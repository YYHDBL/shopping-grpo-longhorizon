"""Reproducibility tests for the pinned veRL source patch."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from verl import DataProto

from scripts import apply_verl_dynamic_sampling_patch as patcher
from shopping_grpo.verl_dynamic_sampling import select_reward_varying_groups


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


class VerlPatchScriptTest(unittest.TestCase):
    def run_script(self, target: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--target", str(target), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_apply_is_idempotent_and_restore_recovers_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)

            first = self.run_script(target)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)
            self.assertIn(patcher.PATCH_MARKER, target.read_text(encoding="utf-8"))

            second = self.run_script(target)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already applied", second.stdout)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)

            restored = self.run_script(target, "--restore")
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_ORIGINAL_SHA256)

    def test_unknown_sha256_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            target.write_text(target.read_text(encoding="utf-8") + "\n# unknown change\n", encoding="utf-8")
            before = file_sha256(target)

            result = self.run_script(target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to patch unknown ray_trainer.py", result.stderr)
            self.assertEqual(file_sha256(target), before)
            self.assertFalse(Path(str(target) + patcher.BACKUP_SUFFIX).exists())

    def test_patched_fit_preserves_bypass_and_defers_reference_and_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            result = self.run_script(target)
            self.assertEqual(result.returncode, 0, result.stderr)

            fit_source = target.read_text(encoding="utf-8").split("    def fit(self):", 1)[1]
            generation = fit_source.index("generate_sequences(combined_gen_batch)")
            reward_filter = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH")
            ready = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_READY")
            sleep_before_training = fit_source.index(
                "self.checkpoint_manager.sleep_replicas()", ready
            )
            bypass = fit_source.index("apply_bypass_mode", sleep_before_training)
            reference = fit_source.index("self._compute_ref_log_prob(batch)", bypass)
            advantage = fit_source.index("batch = compute_advantage(", reference)
            update = fit_source.index("actor_output = self._update_actor(batch)", advantage)

            self.assertLess(generation, reward_filter)
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
                },
                meta_info={"reward_extra_keys": []},
            )

        selected_batches = []
        for batch in (make_batch(0, "a"), make_batch(8, "b")):
            rewards = batch.batch["rm_scores"].sum(dim=-1).tolist()
            indices, _ = select_reward_varying_groups(
                batch.non_tensor_batch["uid"].tolist(), rewards
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
