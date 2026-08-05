"""Reproducibility tests for the pinned veRL source patch."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from verl import DataProto

from scripts import apply_verl_dynamic_sampling_patch as patcher
from shopping_grpo.training.grpo.dynamic_sampling import (
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/apply_verl_dynamic_sampling_patch.py"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_source() -> Path:
    installed = patcher.resolve_installed_ray_trainer()
    if file_sha256(installed) == patcher.EXPECTED_ORIGINAL_SHA256:
        return installed
    backup = Path(str(installed) + patcher.BACKUP_SUFFIX)
    if file_sha256(backup) != patcher.EXPECTED_ORIGINAL_SHA256:
        raise AssertionError("installed veRL source and its backup are not the pinned original")
    return backup


def original_schemas_source() -> Path:
    installed = patcher.resolve_installed_targets()[1]
    if file_sha256(installed) == patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256:
        return installed
    backup = Path(str(installed) + patcher.BACKUP_SUFFIX)
    if file_sha256(backup) != patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256:
        raise AssertionError("installed veRL schemas and its backup are not the pinned original")
    return backup


class VerlPatchScriptTest(unittest.TestCase):
    def run_script(
        self,
        target: Path,
        schemas_target: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--target",
                str(target),
                "--schemas-target",
                str(schemas_target),
                *args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_apply_is_idempotent_and_restore_recovers_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)

            first = self.run_script(target, schemas_target)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_PATCHED_SCHEMAS_SHA256,
            )
            self.assertIn(patcher.PATCH_MARKER, target.read_text(encoding="utf-8"))
            self.assertIn(
                patcher.STRICT_TOOL_SCHEMA_PATCH_MARKER,
                schemas_target.read_text(encoding="utf-8"),
            )

            second = self.run_script(target, schemas_target)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already applied", second.stdout)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_PATCHED_SCHEMAS_SHA256,
            )

            restored = self.run_script(target, schemas_target, "--restore")
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_ORIGINAL_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            )

    def test_unknown_sha256_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# unknown change\n",
                encoding="utf-8",
            )
            before = file_sha256(target)

            result = self.run_script(target, schemas_target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to patch unknown ray_trainer.py", result.stderr)
            self.assertEqual(file_sha256(target), before)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            )
            self.assertFalse(Path(str(target) + patcher.BACKUP_SUFFIX).exists())
            self.assertFalse(Path(str(schemas_target) + patcher.BACKUP_SUFFIX).exists())

    def test_apply_failure_rolls_back_both_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            real_apply_file_section = patcher._apply_file_section

            def fail_on_schemas(patch_program, patch_target, relative_path, temporary):
                if relative_path == patcher.SCHEMAS_RELATIVE_PATH:
                    raise RuntimeError("injected schemas patch failure")
                return real_apply_file_section(
                    patch_program,
                    patch_target,
                    relative_path,
                    temporary,
                )

            with mock.patch.object(
                patcher,
                "_apply_file_section",
                side_effect=fail_on_schemas,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected schemas patch failure"):
                    patcher.apply_patch(target, schemas_target)

            self.assertEqual(file_sha256(target), patcher.EXPECTED_ORIGINAL_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            )
            self.assertEqual(
                file_sha256(Path(str(target) + patcher.BACKUP_SUFFIX)),
                patcher.EXPECTED_ORIGINAL_SHA256,
            )
            self.assertEqual(
                file_sha256(Path(str(schemas_target) + patcher.BACKUP_SUFFIX)),
                patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            )

    def test_apply_migrates_old_trainer_only_patch_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary = Path(temp_dir)
            target = temporary / "ray_trainer.py"
            schemas_target = temporary / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            patcher._ensure_backup(target, patcher.EXPECTED_ORIGINAL_SHA256)
            patch_program = shutil.which("patch")
            self.assertIsNotNone(patch_program)
            patcher._apply_file_section(
                patch_program,
                target,
                patcher.RAY_TRAINER_RELATIVE_PATH,
                temporary,
            )
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            )

            patcher.apply_patch(target, schemas_target)
            patcher.verify_patched(target, schemas_target)
            patcher.restore_patch(target, schemas_target)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_ORIGINAL_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            )

    def test_restore_failure_rolls_back_both_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            patcher.apply_patch(target, schemas_target)

            real_copy2 = shutil.copy2
            schemas_backup = Path(str(schemas_target) + patcher.BACKUP_SUFFIX)

            def fail_on_schemas_restore(source, destination, *args, **kwargs):
                if (
                    Path(source) == schemas_backup
                    and Path(destination).name.endswith(".shopping-grpo-restore.tmp")
                ):
                    raise OSError("injected schemas restore failure")
                return real_copy2(source, destination, *args, **kwargs)

            with mock.patch.object(
                patcher.shutil,
                "copy2",
                side_effect=fail_on_schemas_restore,
            ):
                with self.assertRaisesRegex(OSError, "injected schemas restore failure"):
                    patcher.restore_patch(target, schemas_target)

            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)
            self.assertEqual(
                file_sha256(schemas_target),
                patcher.EXPECTED_PATCHED_SCHEMAS_SHA256,
            )

    def test_unknown_schemas_sha256_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            schemas_target.write_text(
                schemas_target.read_text(encoding="utf-8") + "\n# unknown change\n",
                encoding="utf-8",
            )
            before = file_sha256(schemas_target)

            result = self.run_script(target, schemas_target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to patch unknown schemas.py", result.stderr)
            self.assertEqual(file_sha256(schemas_target), before)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_ORIGINAL_SHA256)
            self.assertFalse(Path(str(target) + patcher.BACKUP_SUFFIX).exists())
            self.assertFalse(Path(str(schemas_target) + patcher.BACKUP_SUFFIX).exists())

    def test_schema_constraints_survive_validation_and_model_dump(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            result = self.run_script(target, schemas_target)
            self.assertEqual(result.returncode, 0, result.stderr)

            spec = importlib.util.spec_from_file_location(
                "patched_verl_tool_schemas",
                schemas_target,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            source_parameters = {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                            "minItems": 1,
                            "maxItems": 2,
                            "uniqueItems": True,
                        },
                        "minItems": 1,
                        "maxItems": 4,
                        "uniqueItems": True,
                    },
                    "explanation": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                "required": ["groups"],
                "additionalProperties": False,
            }
            validated = module.OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "strict_tool",
                        "description": "Schema round-trip regression test.",
                        "parameters": source_parameters,
                    },
                }
            )
            dumped_parameters = validated.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )["function"]["parameters"]
            self.assertEqual(dumped_parameters, source_parameters)

    def test_patched_fit_preserves_bypass_and_defers_reference_and_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            schemas_target = Path(temp_dir) / "schemas.py"
            shutil.copy2(original_source(), target)
            shutil.copy2(original_schemas_source(), schemas_target)
            result = self.run_script(target, schemas_target)
            self.assertEqual(result.returncode, 0, result.stderr)

            fit_source = target.read_text(encoding="utf-8").split("    def fit(self):", 1)[1]
            generation = fit_source.index("generate_sequences(combined_gen_batch)")
            reward_filter = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH")
            skipped = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_SKIPPED")
            ready = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_READY")
            sleep_before_training = fit_source.index(
                "self.checkpoint_manager.sleep_replicas()", ready
            )
            bypass = fit_source.index("apply_bypass_mode", sleep_before_training)
            reference = fit_source.index("self._compute_ref_log_prob(batch)", bypass)
            advantage = fit_source.index("batch = compute_advantage(", reference)
            update = fit_source.index("actor_output = self._update_actor(batch)", advantage)

            self.assertLess(generation, reward_filter)
            self.assertLess(reward_filter, skipped)
            self.assertLess(reward_filter, ready)
            self.assertLess(ready, sleep_before_training)
            self.assertLess(sleep_before_training, bypass)
            self.assertLess(bypass, reference)
            self.assertLess(reference, advantage)
            self.assertLess(advantage, update)
            self.assertIn(
                "if not dynamic_sampling_enabled:\n"
                "                            self.checkpoint_manager.sleep_replicas()",
                fit_source,
            )
            self.assertIn(
                "if bypass_recomputing_logprobs:  # Use `rollout_log_probs`",
                fit_source,
            )
            self.assertIn(
                "else:  # Recompute old_log_probs\n"
                "                        with marked_timer(\"old_log_prob\"",
                fit_source,
            )
            self.assertIn("extract_shopping_group_signals", fit_source)
            self.assertIn("aggregate_shopping_metrics", fit_source)
            self.assertIn("terminal_utilities=terminal_utilities", fit_source)
            self.assertIn("sampling_invalid=sampling_invalid", fit_source)
            self.assertIn('"drop_reason": group["drop_reason"]', fit_source)
            self.assertIn('"group/all_zero_utility_ratio"', fit_source)
            self.assertIn('"group/no_purchase_success_ratio"', fit_source)
            self.assertIn('"group/all_purchase_success_ratio"', fit_source)
            self.assertIn('"group/sampling_invalid"', fit_source)
            self.assertIn('"training/optimizer_updated": 0', fit_source)
            self.assertIn(
                "logger.log(data=skipped_metrics, step=self.global_steps)",
                fit_source,
            )
            self.assertIn(
                "dynamic_accepted_batches = []",
                fit_source[skipped:ready],
            )
            self.assertIn("dynamic_consecutive_skips", fit_source)
            self.assertIn(">= dynamic_max_consecutive_skips", fit_source)
            self.assertNotIn(
                "exhausted max_num_gen_batches=",
                fit_source,
            )
            self.assertLess(
                fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_SKIPPED"),
                fit_source.index("self.checkpoint_manager.sleep_replicas()", ready),
            )

    def test_select_and_concat_keep_all_trajectory_fields_aligned(self):
        def make_batch(offset: int, uid_prefix: str) -> DataProto:
            row_ids = torch.arange(offset, offset + 8, dtype=torch.int64)
            uids = np.array([f"{uid_prefix}-drop"] * 4 + [f"{uid_prefix}-keep"] * 4)
            rewards = torch.tensor([0.0] * 4 + [2 / 7, 4 / 7, 2 / 7, 2 / 7])
            return DataProto.from_dict(
                tensors={
                    "responses": row_ids[:, None],
                    "response_mask": row_ids[:, None],
                    "rollout_log_probs": row_ids[:, None].float(),
                    "token_level_scores": rewards[:, None],
                    "rm_scores": rewards[:, None],
                    "attention_mask": row_ids[:, None],
                    "position_ids": row_ids[:, None],
                },
                non_tensors={
                    "uid": uids,
                    "extra_info": np.array(
                        [{"task_id": int(row_id)} for row_id in row_ids], dtype=object
                    ),
                    "shopping": np.array(
                        [
                            {
                                "infrastructure_invalid": False,
                                "reward": {
                                    "terminal_utility": float(reward),
                                    "purchase_success": bool(reward > 0),
                                    "sampling_invalid": False,
                                },
                            }
                            for reward in rewards
                        ],
                        dtype=object,
                    ),
                },
                meta_info={"reward_extra_keys": []},
            )

        selected_batches = []
        for batch in (make_batch(0, "a"), make_batch(8, "b")):
            rewards = batch.batch["rm_scores"].sum(dim=-1).tolist()
            utility, success, invalid, reasons = extract_shopping_group_signals(
                batch.non_tensor_batch["shopping"].tolist()
            )
            indices, _ = select_reward_varying_groups(
                batch.non_tensor_batch["uid"].tolist(),
                rewards,
                terminal_utilities=utility,
                purchase_success=success,
                sampling_invalid=invalid,
                sampling_invalid_reasons=reasons,
            )
            selected_batches.append(batch.select_idxs(indices))

        combined = DataProto.concat(selected_batches)
        expected_ids = [4, 5, 6, 7, 12, 13, 14, 15]
        self.assertEqual(combined.batch["responses"].flatten().tolist(), expected_ids)
        for key in (
            "response_mask",
            "rollout_log_probs",
            "attention_mask",
            "position_ids",
        ):
            self.assertEqual(combined.batch[key].flatten().tolist(), expected_ids)
        self.assertEqual(
            [item["task_id"] for item in combined.non_tensor_batch["extra_info"]],
            expected_ids,
        )
        self.assertEqual(
            combined.non_tensor_batch["uid"].tolist(),
            ["a-keep"] * 4 + ["b-keep"] * 4,
        )
        self.assertTrue(
            torch.allclose(
                combined.batch["token_level_scores"].flatten(),
                torch.tensor([2 / 7, 4 / 7, 2 / 7, 2 / 7] * 2),
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
