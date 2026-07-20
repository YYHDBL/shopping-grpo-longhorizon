"""验证 benchmark 清单与评测入口的最小 CLI 行为。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.create_shop_benchmark import create_benchmark
from scripts.evaluate_shop_benchmark import parse_args


class BenchmarkCliTest(unittest.TestCase):
    def test_create_benchmark_reserves_tasks_outside_sft(self):
        """写出的清单与元数据必须明确记录 SFT 排除集和固定随机种子。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks = root / "tasks.jsonl"
            sft = root / "sft.jsonl"
            manifest = root / "benchmark.jsonl"
            metadata = root / "metadata.json"
            tasks.write_text("".join(json.dumps({"task_id": task_id}) + "\n" for task_id in range(10)))
            sft.write_text("".join(json.dumps({"task_id": task_id}) + "\n" for task_id in (1, 2)))

            result = create_benchmark(tasks, sft, manifest, metadata, size=4, seed=7)

            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(result["task_count"], 4)
            self.assertFalse({row["task_id"] for row in rows} & {1, 2})
            self.assertEqual(json.loads(metadata.read_text())["seed"], 7)

    def test_evaluation_defaults_match_frozen_protocol(self):
        """Base、SFT、GRPO 必须默认使用同一 35 步上限。"""
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/benchmarks/shop_benchmark_v1.jsonl",
                "--output",
                "outputs/eval/base/raw.jsonl",
                "--summary",
                "outputs/eval/base/summary.json",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.temperature, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
