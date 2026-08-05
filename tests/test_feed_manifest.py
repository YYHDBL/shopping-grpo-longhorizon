import json
from pathlib import Path
import tempfile
import unittest

from shopping_grpo.feed.manifest import (
    audit_split_isolation,
    build_manifest,
    verify_manifest,
    write_json_atomic,
)


def row(episode, persona):
    return {"episode_id": episode, "persona": {"persona_id": persona}}


class FeedManifestTest(unittest.TestCase):
    def test_split_guard_checks_episode_and_persona(self):
        summary = audit_split_isolation(
            {"train": [row("e1", "u1")], "test": [row("e2", "u2")]}
        )
        self.assertEqual(summary["train"]["episodes"], 1)
        with self.assertRaisesRegex(ValueError, "persona_id"):
            audit_split_isolation(
                {"train": [row("e1", "u1")], "test": [row("e2", "u1")]}
            )

    def test_manifest_hashes_are_reopenable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.jsonl").write_text('{"episode_id":"e1"}\n', encoding="utf-8")
            manifest = build_manifest(
                output_dir=root,
                config={"episodes": 1},
                split_summary={"train": {"rows": 1}},
            )
            write_json_atomic(root / "manifest.json", manifest)
            reopened = verify_manifest(root)
            self.assertEqual(reopened["files"]["data.jsonl"]["bytes"], 20)
            payload = json.loads((root / "manifest.json").read_text())
            self.assertEqual(payload["profile_versions"]["environment"], "feed-environment-v1")


if __name__ == "__main__":
    unittest.main()
