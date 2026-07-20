"""验证 SFT 训练样本的 assistant-only loss mask。"""

import unittest
import json
import tempfile
from pathlib import Path

from shopping_grpo.sft_training import (
    IGNORE_INDEX,
    build_supervised_example,
    load_supervised_examples,
    normalize_messages_for_chat_template,
    split_rows_by_task,
)


class CharacterTokenizer:
    """无需 transformers 的最小 chat-template tokenizer，用于验证标签边界。"""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message.get("content") or ""
            for call in message.get("tool_calls") or []:
                text += f"[tool={call['function']['name']}]"
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    @staticmethod
    def decode(token_ids):
        return "".join(chr(token) for token in token_ids)


class DivergentGenerationPromptTokenizer(CharacterTokenizer):
    """模拟部分 Qwen 模板中 generation prompt 与实际 assistant 起始略有差异。"""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        text = super().apply_chat_template(
            messages, tools=tools, tokenize=tokenize, add_generation_prompt=False
        )
        return text + "<assistant>\n" if add_generation_prompt else text


class ProcessorTemplate:
    """模拟 Qwen3.5 的 processor：模板在 processor，分词器在其 tokenizer 属性。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        self.calls.append((messages, tools, tokenize, add_generation_prompt))
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class SftTrainingTest(unittest.TestCase):
    def test_labels_include_only_assistant_content_and_tool_calls(self):
        """user 与 tool observation 必须 mask，assistant 工具调用必须可训练。"""
        messages = [
            {"role": "system", "content": "system rule"},
            {"role": "user", "content": "buy a pillow"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"pillow"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "search result: item-1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "buy_now", "arguments": "{}"}}],
            },
        ]

        example = build_supervised_example(
            messages=messages,
            tools=[{"type": "function"}],
            tokenizer=CharacterTokenizer(),
            max_length=1_000,
        )

        self.assertIsNotNone(example)
        labeled = CharacterTokenizer.decode(
            token for token in example["labels"] if token != IGNORE_INDEX
        )
        self.assertIn("[tool=search_products]", labeled)
        self.assertIn("[tool=buy_now]", labeled)
        self.assertNotIn("buy a pillow", labeled)
        self.assertNotIn("search result: item-1", labeled)

    def test_overlong_trajectory_is_dropped_instead_of_truncated(self):
        """长轨迹不能截断后保留，否则可能只留下半个 tool call。"""
        example = build_supervised_example(
            messages=[
                {"role": "user", "content": "x" * 100},
                {"role": "assistant", "content": "buy"},
            ],
            tools=[],
            tokenizer=CharacterTokenizer(),
            max_length=20,
        )

        self.assertIsNone(example)

    def test_small_generation_prompt_difference_does_not_drop_assistant_turn(self):
        """模板生成提示多一个换行时，仍应通过公共前缀定位 assistant 标签。"""
        example = build_supervised_example(
            messages=[
                {"role": "user", "content": "buy"},
                {"role": "assistant", "content": "purchase"},
            ],
            tools=[],
            tokenizer=DivergentGenerationPromptTokenizer(),
            max_length=1_000,
        )

        self.assertIsNotNone(example)
        labeled = CharacterTokenizer.decode(
            token for token in example["labels"] if token != IGNORE_INDEX
        )
        self.assertIn("purchase", labeled)

    def test_processor_can_own_chat_template_while_tokenizer_encodes_labels(self):
        """Qwen3.5 的 processor 模板必须用于渲染，而 labels 仍由底层 tokenizer 编码。"""
        tokenizer = CharacterTokenizer()
        processor = ProcessorTemplate(tokenizer)
        example = build_supervised_example(
            messages=[
                {"role": "user", "content": "buy"},
                {"role": "assistant", "content": "purchase"},
            ],
            tools=[],
            tokenizer=tokenizer,
            chat_template=processor,
            max_length=1_000,
        )

        self.assertIsNotNone(example)
        self.assertEqual(len(processor.calls), 3)

    def test_openai_tool_call_json_string_is_normalized_for_qwen_template(self):
        """Qwen3.5 模板要求参数是 mapping，原始 OpenAI messages 的 JSON 字符串不能直接传入。"""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"pillow"}',
                        },
                    }
                ],
            }
        ]

        normalized = normalize_messages_for_chat_template(messages)

        self.assertEqual(
            normalized[0]["tool_calls"][0]["function"]["arguments"],
            {"query": "pillow"},
        )
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            '{"query":"pillow"}',
        )

    def test_loader_keeps_valid_rows_and_reports_dropped_rows(self):
        """训练前必须看得到 JSONL 中哪些行不能被目标模板训练。"""
        rows = [
            {
                "task_id": 1,
                "tools": [],
                "messages": [
                    {"role": "user", "content": "buy"},
                    {"role": "assistant", "content": "done"},
                ],
            },
            {
                "task_id": 2,
                "tools": [],
                "messages": [{"role": "user", "content": "no assistant"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sft.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            examples, stats = load_supervised_examples(
                source, tokenizer=CharacterTokenizer(), max_length=1_000
            )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["task_id"], 1)
        self.assertEqual(stats, {"total": 2, "kept": 1, "dropped": 1})

    def test_split_keeps_same_task_out_of_both_train_and_validation(self):
        """同一 task 的多次轨迹不能跨 train/validation，避免评估泄漏。"""
        rows = [
            {"task_id": task_id, "trajectory_id": f"{task_id}-{attempt}"}
            for task_id in range(30)
            for attempt in range(2)
        ]

        train_rows, validation_rows = split_rows_by_task(
            rows, validation_ratio=0.2, seed=42
        )

        train_ids = {row["task_id"] for row in train_rows}
        validation_ids = {row["task_id"] for row in validation_rows}
        self.assertTrue(validation_rows)
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(train_rows) + len(validation_rows), len(rows))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
