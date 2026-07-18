"""验证一键采集并构造 SFT 数据的批次命令。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_sft_batch import batch_paths, parse_args


class CollectSftBatchCliTest(unittest.TestCase):
    def test_defaults_target_a_resumable_hundred_task_batch(self):
        """不传额外参数时，100 条采集使用独立目录和 50 步上限。"""
        with patch.object(sys, "argv", ["collect_sft_batch.py"]):
            args = parse_args()

        self.assertEqual(args.limit, 100)
        self.assertEqual(args.max_steps, 50)
        self.assertEqual(args.output_dir, Path("outputs/collection_100"))

    def test_batch_paths_keep_raw_and_sft_derivatives_together(self):
        """同一批次的原始、验收和训练数据必须落在同一目录。"""
        paths = batch_paths(Path("outputs/example"))

        self.assertEqual(paths["raw"], Path("outputs/example/raw.jsonl"))
        self.assertEqual(paths["accepted"], Path("outputs/example/accepted.jsonl"))
        self.assertEqual(paths["sft"], Path("outputs/example/sft.jsonl"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
