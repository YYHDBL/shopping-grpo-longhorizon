"""验证 LoRA SFT 入口的关键默认值。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_lora_sft import (
    DEFAULT_TARGET_MODULES,
    _load_preprocessing_components,
    parse_args,
)


class _FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _FakeAutoConfig:
    @staticmethod
    def from_pretrained(model_name, trust_remote_code):
        del model_name, trust_remote_code
        return _FakeConfig("qwen3_5")


class _FakeTokenizer:
    pass


class _FakeAutoTokenizer:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeTokenizer()


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeAutoProcessor:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeProcessor()


class TrainLoraSftCliTest(unittest.TestCase):
    def test_defaults_are_suitable_for_small_qwen_lora_warmup(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "/models/Qwen3.5-0.8B",
                "--train",
                "outputs/batch/train.jsonl",
                "--output",
                "checkpoints/qwen-shopping-lora",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.model, "/models/Qwen3.5-0.8B")
        self.assertEqual(args.train, Path("outputs/batch/train.jsonl"))
        self.assertEqual(args.max_length, 24576)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertEqual(args.gradient_accumulation_steps, 8)

    def test_qwen35_uses_processor_template_and_underlying_tokenizer(self):
        """Qwen3.5 是多模态检查点，不能只加载 AutoTokenizer。"""
        tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
            "Qwen/Qwen3.5-2B",
            auto_config=_FakeAutoConfig,
            auto_tokenizer=_FakeAutoTokenizer,
            auto_processor=_FakeAutoProcessor,
        )

        self.assertTrue(is_multimodal)
        self.assertIs(chat_template.tokenizer, tokenizer)
        self.assertTrue(_FakeAutoProcessor.called)
        self.assertFalse(_FakeAutoTokenizer.called)

    def test_default_lora_targets_cover_qwen35_linear_attention_layers(self):
        """Qwen3.5 的 3/4 层是 Gated DeltaNet，不能只训练少数全注意力层。"""
        self.assertIn("in_proj_qkv", DEFAULT_TARGET_MODULES)
        self.assertIn("out_proj", DEFAULT_TARGET_MODULES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
