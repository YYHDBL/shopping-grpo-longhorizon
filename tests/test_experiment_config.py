import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_experiment import build_experiment, load_registry, resolve_experiment


ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTest(unittest.TestCase):
    def test_registered_baselines_preserve_current_training_values(self):
        registry = load_registry(ROOT / "configs/experiments.json")

        sft = resolve_experiment(registry, "sft_baseline")
        grpo = resolve_experiment(registry, "grpo_baseline")

        self.assertEqual(
            {key: sft["settings"][key] for key in ("learning_rate", "epochs", "lora_rank", "lora_alpha")},
            {"learning_rate": 1e-4, "epochs": 3, "lora_rank": 16, "lora_alpha": 32},
        )
        self.assertEqual(sft["settings"]["target_modules"], "full")
        self.assertEqual(
            {
                key: grpo["settings"][key]
                for key in (
                    "learning_rate",
                    "rollout_number",
                    "prompt_batch_size",
                    "ppo_micro_batch_size",
                    "clip_mode",
                    "kl_enabled",
                    "dynamic_sampling",
                    "max_environment_steps",
                )
            },
            {
                "learning_rate": 1e-6,
                "rollout_number": 4,
                "prompt_batch_size": 2,
                "ppo_micro_batch_size": 1,
                "clip_mode": "symmetric",
                "kl_enabled": False,
                "dynamic_sampling": True,
                "max_environment_steps": 35,
            },
        )

    def test_named_and_cli_overrides_translate_to_existing_entrypoints(self):
        registry = load_registry(ROOT / "configs/experiments.json")
        experiment = resolve_experiment(
            registry,
            "grpo_clip_higher",
            ["rollout_number=2", "ppo_mini_batch_size=4", "ppo_micro_batch_size=2"],
        )

        command, environment, output = build_experiment(
            experiment,
            root=ROOT,
            output_root=Path("outputs/ablations"),
        )

        self.assertEqual(output, ROOT / "outputs/ablations/grpo_clip_higher")
        self.assertIn("scripts/train_grpo.py", command)
        self.assertIn("actor_rollout_ref.rollout.n=2", command)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=4", command)
        self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2", command)
        self.assertIn("actor_rollout_ref.actor.clip_ratio_low=0.2", command)
        self.assertIn("actor_rollout_ref.actor.clip_ratio_high=0.28", command)
        self.assertEqual(environment["SHOPPING_LENGTH_SHAPING_ENABLE"], "false")

    def test_trace_arm_only_enables_trace_override(self):
        registry = load_registry(ROOT / "configs/experiments.json")
        baseline, _, _ = build_experiment(
            resolve_experiment(registry, "grpo_baseline"), root=ROOT
        )
        treatment, _, _ = build_experiment(
            resolve_experiment(registry, "grpo_trace"), root=ROOT
        )

        self.assertIn("shopping_trace.enable=false", baseline)
        self.assertIn("shopping_trace.enable=true", treatment)

    def test_external_json_can_add_an_experiment_without_code_changes(self):
        source = json.loads((ROOT / "configs/experiments.json").read_text(encoding="utf-8"))
        source["experiments"]["sft_epoch_4"] = {
            "stage": "sft",
            "overrides": {"epochs": 4, "save_total_limit": 4},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            experiment = resolve_experiment(load_registry(path), "sft_epoch_4")

        self.assertEqual(experiment["settings"]["epochs"], 4)
        self.assertEqual(experiment["settings"]["save_total_limit"], 4)

    def test_dynamic_sampling_and_max_steps_cannot_be_disabled_by_ablation(self):
        registry = load_registry(ROOT / "configs/experiments.json")
        with self.assertRaisesRegex(ValueError, "dynamic_sampling"):
            resolve_experiment(registry, "grpo_baseline", ["dynamic_sampling=false"])
        with self.assertRaisesRegex(ValueError, "max_environment_steps"):
            resolve_experiment(registry, "grpo_baseline", ["max_environment_steps=36"])


if __name__ == "__main__":
    unittest.main()
