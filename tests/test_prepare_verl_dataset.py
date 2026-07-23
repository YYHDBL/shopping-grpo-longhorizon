"""验证 veRL 数据行只暴露用户需求与 task_id，不混入环境隐藏评分信息。"""

import unittest

from scripts.prepare_verl_grpo_dataset import build_verl_record


class PrepareVerlDatasetTest(unittest.TestCase):
    def test_record_uses_visible_instruction_and_top_level_task_id(self):
        row = build_verl_record(7, "Find a blue mug under $20.", split="train", index=3)

        self.assertEqual(row["prompt"][-1], {"role": "user", "content": "Find a blue mug under $20."})
        self.assertEqual(row["extra_info"]["task_id"], 7)
        self.assertNotIn("interaction_kwargs", row["extra_info"])
        serialized = str(row)
        self.assertNotIn("reward_detail", serialized)
        self.assertNotIn("goal", serialized)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
