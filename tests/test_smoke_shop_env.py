import json
import tempfile
import unittest
from pathlib import Path

from scripts import smoke_shop_env


class SmokeShopEnvTest(unittest.TestCase):
    def test_build_url_for_search_action(self):
        url = smoke_shop_env.build_action_url(
            "http://127.0.0.1:7001",
            "fixed_0",
            "search[乳胶枕]",
        )

        self.assertEqual(
            url,
            "http://127.0.0.1:7001/search_results/fixed_0/%5B%27%E4%B9%B3%E8%83%B6%E6%9E%95%27%5D/1",
        )

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


if __name__ == "__main__":
    unittest.main()
