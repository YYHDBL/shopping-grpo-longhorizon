import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shopping_grpo.feed.demo import build_demo_payload, render_demo_html, write_demo


def _row():
    return {
        "episode_id": "ep-1",
        "policy": "rule",
        "episode_return": 1.25,
        "transitions": [
            {
                "observation": {"current_video": {"caption": "desk setup"}},
                "action": {"decision": "no_recommend", "surface": "none"},
                "events": [{"event_type": "watch", "step": 0, "source_step": 0}],
                "reward": {"total": 0.02},
            }
        ],
    }


class FeedDemoTest(unittest.TestCase):
    def test_html_is_self_contained_and_interactive(self):
        source = _row()
        source["behavior_policy"] = source.pop("policy")
        rendered = render_demo_html(build_demo_payload([source]))
        self.assertIn("Feed Agent Lab", rendered)
        self.assertIn('id="step"', rendered)
        self.assertIn("ep-1", rendered)
        self.assertNotIn("https://", rendered)
        self.assertNotIn("purchase_intent", rendered)

    def test_write_demo_reads_jsonl(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs.jsonl"
            logs.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
            destination = write_demo(logs, root / "demo" / "index.html")
            self.assertTrue(destination.is_file())
            self.assertIn("application/json", destination.read_text(encoding="utf-8"))
            manifest = json.loads(
                destination.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["network_dependencies"])
            self.assertEqual(len(manifest["output"]["sha256"]), 64)

    def test_empty_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_demo_payload([])


if __name__ == "__main__":
    unittest.main()
