import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TASK_IDS = {
    2352, 3368, 4071, 4786, 4925, 5475, 5510, 5703, 5904,
    6143, 6520, 11168, 11773, 12860, 19180, 20345, 21785,
}


class EvaluationDatasetTests(unittest.TestCase):
    def test_final_183_manifest_and_blind_guard_match(self):
        tasks_path = ROOT / "data/evaluation/tasks.jsonl"
        task_bytes = tasks_path.read_bytes()
        task_ids = [json.loads(line)["task_id"] for line in task_bytes.decode().splitlines()]
        metadata = json.loads((ROOT / "data/evaluation/metadata.json").read_text())
        guarded = json.loads(
            (ROOT / "src/shopping_grpo/resources/blind_final_task_ids.json").read_text()
        )
        guard = json.loads(
            (ROOT / "src/shopping_grpo/resources/blind_guard.json").read_text()
        )

        self.assertEqual(len(task_ids), 183)
        self.assertEqual(len(set(task_ids)), 183)
        self.assertFalse(EXCLUDED_TASK_IDS.intersection(task_ids))
        self.assertEqual(metadata["name"], "Final-183")
        self.assertEqual(metadata["tasks"], 183)
        self.assertEqual(metadata["sha256"], hashlib.sha256(task_bytes).hexdigest())
        self.assertEqual(guarded["task_ids"], task_ids)
        self.assertEqual(guard["task_count"], 183)
        self.assertEqual(guard["task_sha256"], metadata["sha256"])
        self.assertEqual(
            guard["metadata_sha256"],
            hashlib.sha256((ROOT / "data/evaluation/metadata.json").read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
