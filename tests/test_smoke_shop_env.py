import json
import inspect
import tempfile
import unittest
from pathlib import Path

from scripts import smoke_shop_env


class FakeEnv:
    instances = []

    def __init__(self, base_url):
        self.base_url = base_url
        self.released = False
        self.actions = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.released = True

    def reset(self, task_id):
        return {"instruction": f"task {task_id}", "env_idx": 1}

    def step(self, action):
        self.actions.append(action)
        return {"done": action == "click[Buy Now]", "reward": 0.0, "instruction": action}


class SmokeShopEnvTest(unittest.TestCase):
    def test_smoke_runner_accepts_structured_environment_factory(self):
        parameters = inspect.signature(smoke_shop_env.run_smoke).parameters
        self.assertIn("env_factory", parameters)

    def test_write_run_result_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = smoke_shop_env.write_run_result(
                output_dir=Path(tmpdir),
                task_id=3,
                base_url="http://127.0.0.1:7001",
                steps=[{"action": "start", "status_code": 200}],
            )

            data = json.loads(path.read_text())

        self.assertEqual(data["task_id"], 3)
        self.assertEqual(data["base_url"], "http://127.0.0.1:7001")
        self.assertEqual(data["steps"][0]["action"], "start")

    def test_smoke_runner_records_terminal_step_and_releases(self):
        FakeEnv.instances = []
        with tempfile.TemporaryDirectory() as tmpdir:
            path = smoke_shop_env.run_smoke(
                base_url="http://shop.test",
                task_id=3,
                actions=["search[乳胶枕]", "click[Buy Now]", "search[ignored]"],
                output_dir=Path(tmpdir),
                env_factory=FakeEnv,
            )
            data = json.loads(path.read_text())

        self.assertEqual([step["action"] for step in data["steps"]], ["search[乳胶枕]", "click[Buy Now]"])
        self.assertTrue(FakeEnv.instances[0].released)


if __name__ == "__main__":
    unittest.main()
