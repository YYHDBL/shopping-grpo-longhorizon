import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_shop_task_ids import write_task_ids


class ExportShopTaskIdsTest(unittest.TestCase):
    def test_write_task_ids_uses_contiguous_environment_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "tasks.jsonl"

            count = write_task_ids(goal_count=3, output_path=output)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(rows, [{"task_id": 0}, {"task_id": 1}, {"task_id": 2}])

    def test_write_task_ids_refuses_to_replace_existing_file_without_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "tasks.jsonl"
            output.write_text('{"task_id": 99}\n', encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_task_ids(goal_count=1, output_path=output)

            write_task_ids(goal_count=1, output_path=output, force=True)

            self.assertEqual(output.read_text(encoding="utf-8"), '{"task_id": 0}\n')


if __name__ == "__main__":
    unittest.main()
