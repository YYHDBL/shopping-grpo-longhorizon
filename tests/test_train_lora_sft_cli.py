"""验证 LoRA SFT 入口的关键默认值。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_lora_sft import parse_args


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
        self.assertEqual(args.max_length, 8192)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertEqual(args.gradient_accumulation_steps, 8)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
