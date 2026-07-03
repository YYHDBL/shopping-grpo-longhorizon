import json
import unittest
from pathlib import Path


class ShopTinyDatasetTest(unittest.TestCase):
    def test_dataset_rows_have_shop_interaction_kwargs(self):
        rows = [
            json.loads(line)
            for line in Path("data/shop_tiny_tasks.jsonl").read_text().splitlines()
            if line.strip()
        ]

        self.assertGreaterEqual(len(rows), 2)
        for index, row in enumerate(rows):
            self.assertEqual(row["extra_info"]["index"], index)
            self.assertEqual(row["extra_info"]["interaction_kwargs"]["name"], "shop")
            self.assertIn("task_id", row["extra_info"]["interaction_kwargs"])
            self.assertEqual(row["prompt"][0]["role"], "system")
            self.assertEqual(row["prompt"][1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
