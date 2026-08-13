import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_comparison_report import build_comparison_data


class ComparisonReportTest(unittest.TestCase):
    def test_discovers_runs_and_computes_reward_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "model-a"
            run.mkdir()
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "completed_tasks": 2,
                        "strict_successes": 1,
                        "strict_success_rate": 0.5,
                        "purchase_successes": 1,
                        "purchase_success_rate": 0.5,
                        "reward_valid_tasks": 2,
                        "reward_valid_rate": 1.0,
                        "average_steps": 3.5,
                        "reward_type_counts": {"gold_purchase": 1, "repeat_loop": 1},
                        "protocol": {"model": "Model A"},
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {"task_id": 1, "final_reward": 1.0},
                {"task_id": 2, "final_reward": -0.5},
            ]
            (run / "trajectories.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            data = build_comparison_data(root)

            self.assertEqual([model["key"] for model in data["models"]], ["model-a"])
            self.assertEqual(data["models"][0]["name"], "Model A")
            self.assertEqual(data["models"][0]["reward"]["mean"], 0.25)
            self.assertEqual(data["models"][0]["reward"]["median"], 0.25)
            self.assertEqual(sum(data["models"][0]["reward"]["histogram"]), 2)


if __name__ == "__main__":
    unittest.main()
