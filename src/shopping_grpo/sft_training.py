"""将标准 OpenAI tool-calling messages 转为 LoRA SFT 所需的 labels。

训练时只计算 assistant token 的 loss；system、user 与 tool observation 都是上下文，
其标签固定为 ``IGNORE_INDEX``。边界完全交给目标模型的 chat template 决定，避免
手写 Qwen 特殊 token 或 tool-call 格式。
"""

import json
import hashlib
from pathlib import Path


IGNORE_INDEX = -100


def _token_ids(tokenizer, text):
    """兼容 Hugging Face tokenizer 与测试用的最小 tokenizer。"""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _common_prefix_length(left, right):
    """返回两个 token 序列的最长公共前缀长度。"""
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def build_supervised_example(messages, tools, tokenizer, max_length=8192):
    """渲染一条轨迹，并只保留 assistant 回合对应的训练标签。

    每个 assistant 回合分别渲染「此前消息 + generation prompt」与「包含该回合的
    消息」，两者的 token 差即为该回合的可训练部分，其中自然包含 tool call。
    任何超长或模板边界不一致样本都会丢弃，不做可能截断工具调用的截断。
    """
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        return None

    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = _token_ids(tokenizer, full_text)
    except Exception:
        return None
    if len(input_ids) > int(max_length):
        return None

    labels = [IGNORE_INDEX] * len(input_ids)
    for index in assistant_indices:
        try:
            prefix_text = tokenizer.apply_chat_template(
                messages[:index],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            through_assistant_text = tokenizer.apply_chat_template(
                messages[: index + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = _token_ids(tokenizer, prefix_text)
            through_assistant_ids = _token_ids(tokenizer, through_assistant_text)
        except Exception:
            return None

        # 部分 chat template 的 generation prompt 与实际 assistant 起始 token 会有
        # 极小差异（例如额外换行）。以公共前缀定位，避免把可用样本误判为损坏。
        start = _common_prefix_length(prefix_ids, through_assistant_ids)
        end = len(through_assistant_ids)
        if start >= end or end > len(input_ids):
            return None
        if input_ids[:end] != through_assistant_ids:
            return None
        labels[start:end] = input_ids[start:end]

    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def load_supervised_examples(path, tokenizer, max_length=8192):
    """读取本仓库生成的 SFT JSONL，并报告被模板拒绝的样本数。"""
    examples = []
    stats = {"total": 0, "kept": 0, "dropped": 0}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["total"] += 1
            try:
                row = json.loads(line)
                example = build_supervised_example(
                    messages=row["messages"],
                    tools=row.get("tools") or [],
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
            except (KeyError, TypeError, json.JSONDecodeError):
                example = None
            if example is None:
                stats["dropped"] += 1
                continue
            example["task_id"] = row.get("task_id")
            example["trajectory_id"] = row.get("trajectory_id")
            examples.append(example)
            stats["kept"] += 1
    return examples, stats


def split_rows_by_task(rows, validation_ratio=0.05, seed=42):
    """按 task_id 稳定划分 SFT 行，避免同题轨迹同时出现在训练和验证中。"""
    ratio = float(validation_ratio)
    if not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    task_ids = {row.get("task_id") for row in rows}
    if ratio == 0 or len(task_ids) < 2:
        return list(rows), []

    def stable_key(task_id):
        value = f"{seed}:{task_id}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    ordered_ids = sorted(task_ids, key=stable_key)
    validation_count = max(1, round(len(ordered_ids) * ratio))
    validation_count = min(validation_count, len(ordered_ids) - 1)
    validation_ids = set(ordered_ids[:validation_count])
    validation_rows = [row for row in rows if row.get("task_id") in validation_ids]
    train_rows = [row for row in rows if row.get("task_id") not in validation_ids]
    return train_rows, validation_rows
