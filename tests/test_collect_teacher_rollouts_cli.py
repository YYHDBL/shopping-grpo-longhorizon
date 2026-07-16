"""验证采集命令的默认轨迹步数。"""

import sys
import unittest
from unittest.mock import patch

from scripts.collect_teacher_rollouts import parse_args


class CollectTeacherRolloutsCliTest(unittest.TestCase):
    def test_default_max_steps_is_thirty(self):
        """单轮采集默认对齐 ShopSimulator 论文的 30 步上限。"""
        with patch.object(sys, "argv", ["collect_teacher_rollouts.py", "--tasks", "tasks.jsonl"]):
            args = parse_args()

        self.assertEqual(args.max_steps, 30)

    def test_thinking_flag_enables_explicit_deepseek_thinking_mode(self):
        with patch.object(
            sys,
            "argv",
            [
                "collect_teacher_rollouts.py",
                "--tasks",
                "tasks.jsonl",
                "--thinking",
                "--reasoning-effort",
                "high",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.thinking)
        self.assertEqual(args.reasoning_effort, "high")


if __name__ == "__main__":
    unittest.main()
