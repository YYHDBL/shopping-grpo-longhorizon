import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shopping_grpo.feed.manifest import build_manifest, write_json_atomic
from shopping_grpo.feed.report import build_frozen_report, evaluate_log_file
from scripts.seal_feed_frozen_run import seal_run


def row(policy, episode, decision="no_recommend", *, policy_field="policy"):
    result = {
        "split": "test",
        "episode_id": episode,
        "persona_id": f"persona-{episode}",
        "transitions": [
            {
                "step": 0,
                "action": {
                    "decision": decision,
                    "product_ids": [],
                    "evidence_ids": [],
                },
                "events": [{"event_type": "watch", "value": 6.0}],
                "reward": {"correct_no_recommend": 0.08, "total": 0.1},
            }
        ],
    }
    result[policy_field] = policy
    return result


class FeedReportTest(unittest.TestCase):
    def test_frozen_report_requires_paired_episode_sets(self):
        with self.assertRaisesRegex(ValueError, "unpaired"):
            build_frozen_report([row("a", "e1"), row("b", "e2")])
        report = build_frozen_report(
            [row("a", "e1"), row("b", "e1", policy_field="behavior_policy")],
        )
        self.assertFalse(report["llm_judge"])
        self.assertEqual(report["episode_ids"], ["e1"])
        self.assertEqual(len(report["report_content_sha256"]), 64)

    def test_log_report_writes_hashed_json_markdown_and_manifest(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs.jsonl"
            logs.write_text(
                "\n".join(json.dumps(item) for item in (row("a", "e1"), row("b", "e1")))
                + "\n",
                encoding="utf-8",
            )
            report = evaluate_log_file(logs, root / "report")
            markdown = (root / "report" / "report.md").read_text(encoding="utf-8")
            manifest = json.loads(
                (root / "report" / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
        self.assertIn("deterministic simulator metrics", markdown)
        self.assertEqual(report["policy_count"], 2)
        self.assertEqual(set(manifest["files"]), {"report.json", "report.md"})

    def test_frozen_report_is_bound_to_manifest_test_logs_and_seeds(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            seeds = root / "seeds" / "test.jsonl"
            logs = root / "mixed_policy_logs" / "test.jsonl"
            seeds.parent.mkdir(parents=True)
            logs.parent.mkdir(parents=True)
            seeds.write_text(
                json.dumps(
                    {"episode_id": "e1", "persona": {"persona_id": "persona-e1"}}
                )
                + "\n",
                encoding="utf-8",
            )
            payload = "\n".join(
                json.dumps(item) for item in (row("a", "e1"), row("b", "e1"))
            ) + "\n"
            logs.write_text(payload, encoding="utf-8")
            manifest = build_manifest(
                output_dir=root,
                config={"episodes": 1},
                split_summary={"test": {"rows": 1}},
                include_paths=("seeds", "mixed_policy_logs"),
            )
            write_json_atomic(root / "manifest.json", manifest)

            report = evaluate_log_file(
                logs,
                root / "evaluation",
                dataset_dir=root,
            )
            copied_logs = root / "copied-test.jsonl"
            copied_logs.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run-manifest"):
                evaluate_log_file(
                    copied_logs,
                    root / "relabelled-evaluation",
                    dataset_dir=root,
                )

            model_logs = root / "model-test.jsonl"
            model_logs.write_text(json.dumps(row("model-v1", "e1")) + "\n", encoding="utf-8")
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"frozen-checkpoint")
            run_manifest = root / "model-test.run.json"
            sealed = seal_run(
                model_logs,
                root,
                checkpoint,
                run_manifest,
                policy_id="model-v1",
            )
            model_report = evaluate_log_file(
                model_logs,
                root / "model-evaluation",
                dataset_dir=root,
                run_manifest=run_manifest,
            )
        self.assertEqual(report["episode_ids"], ["e1"])
        self.assertEqual(model_report["source"]["run_type"], "sealed_model_rollout")
        self.assertEqual(model_report["source"]["policy_id"], "model-v1")
        self.assertEqual(len(sealed["checkpoint"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
