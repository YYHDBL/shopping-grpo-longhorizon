"""验证一键采集并构造 SFT 数据的批次命令。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.collect_sft_batch import _collect_until_target, batch_paths, parse_args


class CollectSftBatchCliTest(unittest.TestCase):
    def test_defaults_target_a_resumable_hundred_task_batch(self):
        """不传额外参数时，100 条采集使用独立目录和 35 步上限。"""
        with patch.object(sys, "argv", ["collect_sft_batch.py"]):
            args = parse_args()

        self.assertEqual(args.limit, 100)
        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.output_dir, Path("outputs/collection_100"))

    def test_batch_paths_keep_raw_and_sft_derivatives_together(self):
        """同一批次的原始、验收和训练数据必须落在同一目录。"""
        paths = batch_paths(Path("outputs/example"))

        self.assertEqual(paths["raw"], Path("outputs/example/raw.jsonl"))
        self.assertEqual(paths["accepted"], Path("outputs/example/accepted.jsonl"))
        self.assertEqual(paths["sft"], Path("outputs/example/sft.jsonl"))

    def test_target_accepted_is_parsed_for_exact_collection_stop(self):
        with patch.object(
            sys,
            "argv",
            ["collect_sft_batch.py", "--limit", "900", "--target-accepted", "500"],
        ):
            args = parse_args()

        self.assertEqual(args.limit, 900)
        self.assertEqual(args.target_accepted, 500)

    def test_workers_is_parsed_for_parallel_collection(self):
        with patch.object(sys, "argv", ["collect_sft_batch.py", "--workers", "4"]):
            args = parse_args()

        self.assertEqual(args.workers, 4)

    def test_target_collection_forwards_workers_to_parallel_scheduler(self):
        """目标模式必须把 CLI worker 数传给并发调度器，不能静默退化为串行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                sys,
                "argv",
                [
                    "collect_sft_batch.py",
                    "--target-accepted",
                    "2",
                    "--workers",
                    "4",
                    "--output-dir",
                    tmpdir,
                    "--llm-base-url",
                    "https://model.test/v1",
                    "--api-key",
                    "test-key",
                ],
            ), patch("scripts.collect_sft_batch.load_tasks", return_value=[]), patch(
                "scripts.collect_sft_batch._collect_until_target", return_value=([], 0)
            ) as collect, patch(
                "scripts.collect_sft_batch._build_derivatives",
                return_value={"total": 0, "accepted": 0, "rejected": 0},
            ), patch("scripts.collect_sft_batch.signal.signal"):
                from scripts.collect_sft_batch import main

                main()

        self.assertEqual(collect.call_args.kwargs["workers"], 4)

    def test_collection_stops_after_target_number_of_accepted_trajectories(self):
        with patch("scripts.collect_sft_batch.collect_for_task") as collect, patch(
            "scripts.collect_sft_batch.acceptance_reasons", return_value=(True, [])
        ), patch(
            "scripts.collect_sft_batch._progress_bar", return_value=MagicMock()
        ):
            collect.side_effect = [
                {"trajectory_id": "first", "task_id": 1},
                {"trajectory_id": "second", "task_id": 2},
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                written, accepted = _collect_until_target(
                    tasks=[{"task_id": 1}, {"task_id": 2}, {"task_id": 3}],
                    target_accepted=2,
                    client=object(),
                    output_path=Path(tmpdir) / "raw.jsonl",
                    base_url="http://shop.test",
                    max_steps=50,
                    attempts_per_task=1,
                    workers=1,
                )

        self.assertEqual([row["trajectory_id"] for row in written], ["first", "second"])
        self.assertEqual(accepted, 2)
        self.assertEqual(collect.call_count, 2)

    def test_parallel_collection_never_exceeds_accepted_target(self):
        rows = [
            {"trajectory_id": f"trajectory-{index}", "task_id": index}
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "scripts.collect_sft_batch.collect_for_task", side_effect=rows
        ) as collect, patch(
            "scripts.collect_sft_batch.acceptance_reasons", return_value=(True, [])
        ), patch(
            "scripts.collect_sft_batch._progress_bar", return_value=MagicMock()
        ):
            written, accepted = _collect_until_target(
                tasks=[{"task_id": index} for index in range(4)],
                target_accepted=2,
                client=object(),
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=50,
                attempts_per_task=1,
                workers=2,
            )

        self.assertEqual(accepted, 2)
        self.assertEqual(len(written), 2)
        self.assertEqual(collect.call_count, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
