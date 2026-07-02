import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.rollout import make_step, make_trajectory, write_jsonl


class RolloutTest(unittest.TestCase):
    def test_make_trajectory_has_required_fields(self):
        traj = make_trajectory(task_id=0, steps=[], final_reward=0.0, done=False)

        self.assertTrue({"task_id", "steps", "final_reward", "done"}.issubset(traj))

    def test_make_step_records_tool_and_action(self):
        step = make_step(
            tool_name="search_products",
            parameters={"query": "乳胶枕"},
            action="search[乳胶枕]",
            observation={"observation": "result"},
            reward=0.0,
            done=False,
            info={"status_code": 200},
        )

        self.assertEqual(step["tool_name"], "search_products")
        self.assertEqual(step["action"], "search[乳胶枕]")

    def test_write_jsonl_writes_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traj.jsonl"
            write_jsonl(path, [make_trajectory(task_id=1, steps=[], final_reward=0.0, done=False)])
            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(rows[0]["task_id"], 1)


if __name__ == "__main__":
    unittest.main()
