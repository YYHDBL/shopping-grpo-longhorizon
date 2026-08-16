import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_glm_report import build_data


class EvaluationReportTest(unittest.TestCase):
    def test_report_reads_any_evaluation_directory_and_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "completed_tasks": 1,
                "done_tasks": 1,
                "done_rate": 1.0,
                "strict_successes": 1,
                "strict_success_rate": 1.0,
                "strict_success_task_ids": [42],
                "purchase_successes": 1,
                "purchase_success_rate": 1.0,
                "mean_final_reward": 1.0,
                "mean_weighted_score": 1.0,
                "average_steps": 1.0,
                "protocol": {"model": "new-model-1"},
            }
            trajectory = {
                "task_id": 42,
                "status": "done",
                "done": True,
                "final_reward": 1.0,
                "steps": [],
                "blocked_tool_calls": [],
                "initial_result": {"instruction": "test"},
                "terminal_result": {
                    "reward_detail": {"reward_type": "gold_purchase", "purchase_success": True},
                    "termination_reason": "gold_purchase",
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "trajectories.jsonl").write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )

            data = build_data(run_dir)

            self.assertEqual(data["meta"]["model"], "new-model-1")
            self.assertEqual(data["summary"]["total"], 1)

    def test_report_includes_early_abstain_and_guard_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "completed_tasks": 1,
                "done_tasks": 1,
                "done_rate": 1.0,
                "strict_successes": 0,
                "strict_success_rate": 0.0,
                "strict_success_task_ids": [],
                "purchase_successes": 0,
                "purchase_success_rate": 0.0,
                "mean_final_reward": 0.0,
                "mean_weighted_score": 0.0,
                "average_steps": 1.0,
                "protocol": {"model": "new-model-1"},
            }
            trajectory = {
                "task_id": 42,
                "status": "done",
                "done": True,
                "final_reward": 0.0,
                "steps": [],
                "blocked_tool_calls": [{"reason": "test_guard"}],
                "initial_result": {"instruction": "test"},
                "terminal_result": {
                    "reward_detail": {
                        "reward_type": "early_abstain",
                        "purchase_success": False,
                    },
                    "termination_reason": "early_abstain",
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "trajectories.jsonl").write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )

            data = build_data(run_dir)

            early_abstain = next(
                item for item in data["charts"]["outcomes"]
                if item["key"] == "early_abstain"
            )
            self.assertEqual(early_abstain["value"], 1)
            self.assertEqual(data["summary"]["guard_per_task"], 1.0)


if __name__ == "__main__":
    unittest.main()
