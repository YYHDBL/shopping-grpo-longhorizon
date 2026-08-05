from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.check_feed_grpo_runtime import (
    DYNAMIC_SAMPLING_PATCH_MARKER,
    static_preflight,
    _validate_feed_parquet_rows,
    validate_tool_schema_roundtrip,
    verify_dynamic_sampling_patch,
)
from scripts.train_feed_grpo import build_command, main, parse_args
from shopping_grpo.feed.observation import render_feed_observation


ROOT = Path(__file__).resolve().parents[1]


def _launcher_argv(root: Path, *extra: str, config: Path | None = None) -> list[str]:
    model = root / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"stub")
    parquet = root / "parquet"
    parquet.mkdir(exist_ok=True)
    train = parquet / "train.parquet"
    validation = parquet / "validation.parquet"
    (parquet / "manifest.json").write_text("{}", encoding="utf-8")
    dataset = root / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    catalog = root / "catalog.jsonl"
    for path in (train, validation, catalog):
        path.write_bytes(b"stub")
    argv = [
        "--model",
        str(model),
        "--train-data",
        str(train),
        "--val-data",
        str(validation),
        "--dataset-dir",
        str(dataset),
        "--catalog",
        str(catalog),
        "--output",
        str(root / "output"),
    ]
    if config is not None:
        argv.extend(("--config", str(config)))
    argv.extend(extra)
    return argv


class _PreservingSchema:
    def __init__(self, value):
        self.value = value

    @classmethod
    def model_validate(cls, value):
        return cls(deepcopy(value))

    def model_dump(self, **kwargs):
        del kwargs
        return deepcopy(self.value)


class _ConstraintDroppingSchema(_PreservingSchema):
    @classmethod
    def model_validate(cls, value):
        copied = deepcopy(value)
        copied["function"]["parameters"].pop("additionalProperties", None)
        return cls(copied)


class FeedGRPOHardeningTest(unittest.TestCase):
    def test_parquet_row_contract_rejects_wrong_length_and_accepts_isolated_identity(self):
        initial = render_feed_observation(
            {
                "observation_version": "feed-observation-v1",
                "environment_version": "feed-environment-v1",
                "episode_id": "e1",
                "step": 0,
                "total_steps": 24,
                "persona": {"persona_id": "p1"},
                "current_video": {"video_id": "v0"},
                "recent_events": [],
                "cart": [],
                "purchased_product_ids": [],
                "visible_product_ids": [],
                "evidence_ids": [],
                "info_tool_calls": 0,
                "max_info_tool_calls": 3,
                "done": False,
            }
        )
        row = {
            "data_source": "feed-shopping-v1",
            "task_id": "e1",
            "prompt": [{"role": "user", "content": initial}],
            "episode_seed": json.dumps(
                {
                    "episode_id": "e1",
                    "persona": {"persona_id": "p1"},
                    "videos": [{"video_id": f"v{i}"} for i in range(24)],
                    "metadata": {"split": "train"},
                }
            ),
            "catalog_products": json.dumps([{"product_id": "x", "title": "item"}]),
        }
        self.assertEqual(
            _validate_feed_parquet_rows([row], split="train"),
            {("e1", "p1")},
        )
        invalid = deepcopy(row)
        seed = json.loads(invalid["episode_seed"])
        seed["videos"] = seed["videos"] * 3
        invalid["episode_seed"] = json.dumps(seed)
        with self.assertRaisesRegex(ValueError, "24--48"):
            _validate_feed_parquet_rows([invalid], split="train")
        override = deepcopy(row)
        override["extra_info"] = {"feed_task": {"episode_seed": {}}}
        with self.assertRaisesRegex(ValueError, "runtime overrides"):
            _validate_feed_parquet_rows([override], split="train")
        mismatch = deepcopy(row)
        mismatch["extra_info"] = {
            "episode_id": "e1",
            "persona_id": "p1",
            "split": "validation",
            "episode_seed": mismatch["episode_seed"],
        }
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            _validate_feed_parquet_rows([mismatch], split="train")
    def test_schema_roundtrip_detects_dropped_strict_constraints(self):
        report = validate_tool_schema_roundtrip(_PreservingSchema)
        self.assertTrue(report["supported"])
        self.assertEqual(len(report["tools_checked"]), 8)
        self.assertGreater(report["constraint_count"], 0)
        with self.assertRaisesRegex(ValueError, "dropped or changed"):
            validate_tool_schema_roundtrip(_ConstraintDroppingSchema)

    def test_patch_marker_is_read_from_imported_trainer_source(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "ray_trainer.py"
            source.write_text(
                f"# {DYNAMIC_SAMPLING_PATCH_MARKER}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                verify_dynamic_sampling_patch(SimpleNamespace(__file__=str(source))),
                str(source.resolve()),
            )
            source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required patch marker"):
                verify_dynamic_sampling_patch(SimpleNamespace(__file__=str(source)))

    def test_preflight_validates_the_exact_custom_grpo_config(self):
        with TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "custom_feed.yaml"
            original = (ROOT / "configs" / "feed_grpo.yaml").read_text(encoding="utf-8")
            custom.write_text(original, encoding="utf-8")
            report = static_preflight(ROOT, grpo_config=custom)
            self.assertEqual(report["config"]["grpo_config"], str(custom.resolve()))

            custom.write_text(
                original.replace("max_response_length: 61440", "max_response_length: 1024"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "context lengths"):
                static_preflight(ROOT, grpo_config=custom)

    def test_protected_hydra_overrides_are_rejected_but_tuning_is_allowed(self):
        protected = (
            "data.max_response_length=1024",
            "+actor_rollout_ref.rollout.multi_turn.tool_config_path=/tmp/other.json",
            "actor_rollout_ref.rollout.agent.agent_loop_config_path=/tmp/other.yaml",
            "shopping_dynamic_sampling.enable=false",
            "reward_model.enable=true",
            "algorithm.adv_estimator=gae",
            "feed_process_credit.metadata_only=false",
            "feed_runtime.llm_judge=true",
            "+foo@actor_rollout_ref/rollout/multi_turn=unsafe_profile",
            "--config-name=other",
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for override in protected:
                with self.subTest(override=override):
                    args = parse_args(_launcher_argv(root, "--", override))
                    with self.assertRaisesRegex(SystemExit, "not allowed|runtime contract"):
                        build_command(args)

            args = parse_args(
                _launcher_argv(
                    root,
                    "--",
                    "actor_rollout_ref.actor.optim.lr=2e-6",
                    "actor_rollout_ref.rollout.temperature=0.8",
                    "trainer.total_training_steps=25",
                )
            )
            command, _, _ = build_command(args)
            self.assertIn("actor_rollout_ref.actor.optim.lr=2e-6", command)
            self.assertIn("actor_rollout_ref.rollout.temperature=0.8", command)

    def test_launcher_rejects_model_with_short_declared_context(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            argv = _launcher_argv(root, "--dry-run")
            (root / "model" / "config.json").write_text(
                json.dumps({"max_position_embeddings": 32768}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "smaller than Feed runtime 65536"):
                build_command(parse_args(argv))

    def test_dry_run_reports_false_without_invoking_a_subprocess(self):
        with TemporaryDirectory() as temp_dir:
            argv = _launcher_argv(Path(temp_dir), "--dry-run")
            output = StringIO()
            with patch("scripts.train_feed_grpo.subprocess.call") as call:
                with redirect_stdout(output):
                    main(argv)
            call.assert_not_called()
            audit = json.loads(output.getvalue())
            self.assertFalse(audit["training_started"])

    def test_actual_launcher_passes_config_and_reports_started_only_at_exec_boundary(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            custom = root / "custom_feed.yaml"
            custom.write_text(
                (ROOT / "configs" / "feed_grpo.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            argv = _launcher_argv(root, config=custom)
            output = StringIO()
            calls: list[list[str]] = []

            def fake_call(command, **kwargs):
                del kwargs
                calls.append(list(command))
                if len(calls) == 1:
                    self.assertNotIn("training_started", output.getvalue())
                else:
                    self.assertIn('"training_started": true', output.getvalue())
                return 0

            with patch("scripts.train_feed_grpo.subprocess.call", side_effect=fake_call):
                with redirect_stdout(output):
                    with self.assertRaises(SystemExit) as stopped:
                        main(argv)
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(len(calls), 2)
            self.assertIn("--require-runtime", calls[0])
            config_index = calls[0].index("--grpo-config") + 1
            self.assertEqual(calls[0][config_index], str(custom.resolve()))
            self.assertIn("--train-data", calls[0])
            self.assertIn("--val-data", calls[0])
            audit = json.loads(output.getvalue())
            self.assertTrue(audit["training_started"])


if __name__ == "__main__":
    unittest.main()
