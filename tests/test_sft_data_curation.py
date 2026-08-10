import unittest

from scripts.label_sft_difficulty import make_batches
from scripts.merge_sft_pure_v4 import _candidate_key


def trajectory(task_id, *, steps, guard=False):
    messages = [{"role": "user", "content": "buy one"}]
    for index in range(steps):
        name = "buy_now" if index == steps - 1 else "search_products"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": name}}],
                },
                {
                    "role": "tool",
                    "content": "guard rejected" if guard else "ok" * 100,
                },
            ]
        )
    return {
        "task_id": task_id,
        "trajectory_id": str(task_id),
        "messages": messages,
        "tools": [],
    }


class SftDataCurationTests(unittest.TestCase):
    def test_quality_priority_and_tool_page_compaction(self):
        clean_longer = trajectory(1, steps=2)
        guarded_shorter = trajectory(1, steps=1, guard=True)
        self.assertGreater(
            _candidate_key(clean_longer, seed=1),
            _candidate_key(guarded_shorter, seed=1),
        )

        clean_shorter = trajectory(1, steps=1)
        self.assertGreater(
            _candidate_key(clean_shorter, seed=1),
            _candidate_key(clean_longer, seed=1),
        )

        batch = make_batches([clean_longer], 1, 10000, max_tool_content_chars=20)
        tool_messages = [
            message
            for message in batch[0][0]["trajectory"]
            if message["role"] == "tool"
        ]
        self.assertTrue(all("truncated" in message["content"] for message in tool_messages))


if __name__ == "__main__":
    unittest.main()
