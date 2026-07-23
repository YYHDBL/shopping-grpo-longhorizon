"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_grpo_runtime import (
    PATCH_MARKER,
    compose_runtime_config,
    validate_dynamic_sampling,
)


class DynamicSamplingConfigTest(unittest.TestCase):
    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = compose_runtime_config(
            [
                "shopping_dynamic_sampling.enable=true",
                "shopping_dynamic_sampling.metric=seq_reward",
                "shopping_dynamic_sampling.max_num_gen_batches=3",
                "shopping_dynamic_sampling.reward_tolerance=1e-8",
            ]
        )
        self.assertTrue(config.shopping_dynamic_sampling.enable)
        self.assertEqual(config.shopping_dynamic_sampling.metric, "seq_reward")
        self.assertEqual(config.shopping_dynamic_sampling.max_num_gen_batches, 3)
        self.assertEqual(config.shopping_dynamic_sampling.reward_tolerance, 1.0e-8)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertTrue(config.actor_rollout_ref.rollout.calculate_log_probs)

    def test_enabled_config_requires_installed_patch_marker(self):
        config = compose_runtime_config(["shopping_dynamic_sampling.enable=true"])
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            trainer_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "patch marker is missing"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            trainer_source.write_text(f"# {PATCH_MARKER}\n", encoding="utf-8")
            validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
