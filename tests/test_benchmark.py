"""验证固定 benchmark 的划分与统计口径。"""

import unittest

from shopping_grpo.benchmark import build_benchmark_manifest, summarize_trajectories


def _trajectory(task_id, strict=False, steps=3, status="done", blocked=None):
    reward_detail = {
        "r_type": 1 if strict else 0,
        "r_att": 1,
        "r_option": 1,
        "r_price": 1,
    }
    return {
        "task_id": task_id,
        "status": status,
        "done": status == "done",
        "final_reward": 1.0 if strict else 0.25,
        "steps": [{"tool_name": "search_products"}] * steps,
        "blocked_tool_calls": blocked or [],
        "terminal_result": {"done": status == "done", "over": status == "done", "reward_detail": reward_detail},
    }


class BenchmarkTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_excludes_sft_tasks(self):
        """benchmark task 必须固定，且不能与冷启动 SFT task 泄漏重合。"""
        first = build_benchmark_manifest(
            all_task_ids=range(20), excluded_task_ids={1, 3, 5}, size=8, seed=20260720
        )
        second = build_benchmark_manifest(
            all_task_ids=range(20), excluded_task_ids={1, 3, 5}, size=8, seed=20260720
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertFalse({row["task_id"] for row in first} & {1, 3, 5})

    def test_summary_uses_expected_tasks_as_strict_success_denominator(self):
        """缺失或非严格成功 task 都计入失败，避免只统计已跑完的容易样本。"""
        summary = summarize_trajectories(
            expected_task_ids=[10, 11, 12],
            trajectories=[
                _trajectory(10, strict=True, steps=4),
                _trajectory(
                    11,
                    strict=False,
                    steps=6,
                    blocked=[{"reason": "schema_extra_arguments:asin"}],
                ),
            ],
        )

        self.assertEqual(summary["expected_tasks"], 3)
        self.assertEqual(summary["completed_tasks"], 2)
        self.assertEqual(summary["strict_successes"], 1)
        self.assertAlmostEqual(summary["strict_success_rate"], 1 / 3)
        self.assertEqual(summary["missing_tasks"], [12])
        self.assertAlmostEqual(summary["average_steps"], 5.0)
        self.assertEqual(summary["guard_reason_counts"]["schema_extra_arguments:asin"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
