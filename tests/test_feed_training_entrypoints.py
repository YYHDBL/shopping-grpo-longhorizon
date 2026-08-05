import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.prepare_feed_grpo_data import inspect_input, normalize_task_for_parquet
from scripts.train_feed_grpo import build_command, parse_args
from scripts.train_lora_sft import _enforce_sample_retention


class FeedTrainingEntrypointsTest(unittest.TestCase):
    def test_strict_sft_retention_rejects_any_tokenizer_drop(self):
        _enforce_sample_retention(
            {"total": 2, "kept": 2, "dropped": 0},
            label="train",
            enabled=True,
        )
        with self.assertRaisesRegex(SystemExit, "1/2"):
            _enforce_sample_retention(
                {"total": 2, "kept": 1, "dropped": 1},
                label="train",
                enabled=True,
            )
    def test_parquet_normalization_keeps_prompt_and_encodes_dynamic_truth(self):
        row = {
            "data_source": "feed-shopping-v1",
            "task_id": "e1",
            "prompt": [{"role": "user", "content": "hello"}],
            "episode_seed": {"episode_id": "e1", "metadata": {"dynamic": {"v1": 1}}},
            "catalog_products": [{"product_id": "P1"}],
            "calibration": {"click_probability": 0.1},
        }
        normalized = normalize_task_for_parquet(row)
        self.assertIsInstance(normalized["prompt"], list)
        self.assertEqual(json.loads(normalized["episode_seed"])["episode_id"], "e1")
        self.assertEqual(json.loads(normalized["catalog_products"])[0]["product_id"], "P1")

    def test_inspect_and_feed_launcher_dry_run_contract(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tasks = root / "train.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "data_source": "feed-shopping-v1",
                        "task_id": "e1",
                        "prompt": [],
                        "episode_seed": {
                            "episode_id": "e1",
                            "persona": {"persona_id": "p1"},
                        },
                        "catalog_products": [{"product_id": "P1"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(inspect_input(tasks)["row_count"], 1)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"stub")
            train = root / "train.parquet"
            validation = root / "validation.parquet"
            catalog = root / "catalog.jsonl"
            for path in (train, validation, catalog):
                path.write_bytes(b"stub")
            args = parse_args(
                [
                    "--model", str(model),
                    "--train-data", str(train),
                    "--val-data", str(validation),
                    "--catalog", str(catalog),
                    "--output", str(root / "output"),
                    "--dry-run",
                ]
            )
            command, environment, audit = build_command(args)
        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["FEED_CREDIT_MODE"], "terminal")
        self.assertFalse(audit["training_started"])
        self.assertFalse(audit["native_advantage_integration"])

    def test_nonterminal_credit_requires_explicit_boundary_acknowledgement(self):
        # The argument-level boundary is tested without constructing file fixtures.
        args = parse_args(
            [
                "--model", "missing",
                "--train-data", "missing",
                "--val-data", "missing",
                "--output", "missing",
                "--credit-mode", "event",
            ]
        )
        self.assertEqual(args.credit_mode, "event")
        self.assertFalse(args.acknowledge_metadata_only)


if __name__ == "__main__":
    unittest.main()
