"""验证 benchmark v2 从历史 v1 精确裁剪，避免重抽造成横向不可比。"""

import unittest

from shopping_grpo.grpo_tasks import freeze_benchmark_subset


class GrpoBenchmarkTest(unittest.TestCase):
    def test_freeze_subset_keeps_existing_order_and_records_parent(self):
        parent = [{"task_id": 9}, {"task_id": 3}, {"task_id": 7}]
        rows, metadata = freeze_benchmark_subset(
            parent, parent_name="shop_benchmark_v1", size=2, parent_sha256="abc"
        )

        self.assertEqual(rows, [{"task_id": 9}, {"task_id": 3}])
        self.assertEqual(metadata["parent_benchmark"], "shop_benchmark_v1")
        self.assertEqual(metadata["parent_sha256"], "abc")
        self.assertEqual(metadata["selection"], "ordered_prefix")

    def test_freeze_subset_rejects_duplicate_or_oversized_input(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze_benchmark_subset([{"task_id": 1}, {"task_id": 1}], "v1", 1, "abc")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            freeze_benchmark_subset([{"task_id": 1}], "v1", 2, "abc")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
