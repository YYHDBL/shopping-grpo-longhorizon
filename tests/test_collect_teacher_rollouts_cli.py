"""验证采集命令的默认轨迹步数。"""

import sys
import unittest
from unittest.mock import patch

from scripts.collect_teacher_rollouts import parse_args


class CollectTeacherRolloutsCliTest(unittest.TestCase):
    def test_default_max_steps_is_sixteen(self):
        """默认值应覆盖一次完整的搜索、核验规格和购买流程。"""
        with patch.object(sys, "argv", ["collect_teacher_rollouts.py", "--tasks", "tasks.jsonl"]):
            args = parse_args()

        self.assertEqual(args.max_steps, 16)


if __name__ == "__main__":
    unittest.main()
