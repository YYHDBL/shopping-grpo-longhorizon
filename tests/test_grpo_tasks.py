"""验证 GRPO 任务划分：训练、评测与历史 SFT 采集必须严格隔离。"""

import unittest

from shopping_grpo.grpo_tasks import (
    build_grpo_candidate_manifest,
    select_disjoint_validation_tasks,
    select_stratified_grpo_tasks,
)


def _probe(task_id, steps, status="done"):
    return {"task_id": task_id, "status": status, "steps": [{}] * steps}


class GrpoTaskSplitTest(unittest.TestCase):
    def test_candidate_pool_excludes_all_historical_rollouts_and_benchmark(self):
        """拒绝轨迹也不能与在线 RL task 重叠，避免隐性题目泄漏。"""
        rows = build_grpo_candidate_manifest(
            all_task_ids=range(20),
            excluded_rollout_task_ids={1, 2, 3},
            benchmark_task_ids={4, 5},
            size=10,
            seed=7,
        )

        selected = {row["task_id"] for row in rows}
        self.assertEqual(len(rows), 10)
        self.assertFalse(selected & {1, 2, 3, 4, 5})
        self.assertEqual(
            rows,
            build_grpo_candidate_manifest(
                range(20), {1, 2, 3}, {4, 5}, size=10, seed=7
            ),
        )

    def test_stratified_selection_uses_policy_probe_lengths_and_skips_infra_errors(self):
        """按 SFT policy 实测步数分层；HTTP/环境错误不应被当作长任务。"""
        candidates = [{"task_id": task_id} for task_id in range(12)]
        probes = [
            _probe(0, 8), _probe(1, 9), _probe(2, 10),
            _probe(3, 11), _probe(4, 15), _probe(5, 20),
            _probe(6, 21), _probe(7, 25), _probe(8, 35, "max_steps"),
            _probe(9, 1, "error"),
        ]

        rows, report = select_stratified_grpo_tasks(
            candidates,
            probes,
            bucket_targets={"short": 2, "medium": 2, "long": 2},
            seed=42,
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["length_bucket"] for row in rows}, {"short", "medium", "long"})
        self.assertEqual(report["eligible_probe_count"], 9)
        self.assertEqual(report["ignored_probe_status_counts"], {"error": 1})
        self.assertNotIn(9, {row["task_id"] for row in rows})

    def test_stratified_selection_fails_instead_of_silently_unbalancing(self):
        """某长度桶不足时必须停下报告，不能偷偷以其他桶补齐。"""
        with self.assertRaisesRegex(ValueError, "long"):
            select_stratified_grpo_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                [_probe(1, 3), _probe(2, 15)],
                bucket_targets={"short": 1, "medium": 0, "long": 1},
                seed=1,
            )

    def test_validation_selection_is_deterministic_and_disjoint_from_train(self):
        candidates = [{"task_id": task_id} for task_id in range(10)]
        train = [{"task_id": 1}, {"task_id": 4}, {"task_id": 8}]

        rows = select_disjoint_validation_tasks(candidates, train, size=4, seed=7)

        self.assertEqual(len(rows), 4)
        self.assertFalse({row["task_id"] for row in rows} & {1, 4, 8})
        self.assertEqual(rows, select_disjoint_validation_tasks(candidates, train, size=4, seed=7))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
