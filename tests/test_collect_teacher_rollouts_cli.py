"""验证采集命令的默认轨迹步数。"""

import sys
import unittest
from unittest.mock import patch

from scripts.collect_teacher_rollouts import parse_args


class CollectTeacherRolloutsCliTest(unittest.TestCase):
    def test_default_max_steps_is_thirty_five(self):
        """默认 35 步，避免长失败轨迹持续消耗采集时间。"""
        with patch.object(sys, "argv", ["collect_teacher_rollouts.py", "--tasks", "tasks.jsonl"]):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)

    def test_default_timeout_allows_long_thinking_tool_calls(self):
        with patch.object(sys, "argv", ["collect_teacher_rollouts.py", "--tasks", "tasks.jsonl"]):
            args = parse_args()

        self.assertEqual(args.timeout, 180)

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
