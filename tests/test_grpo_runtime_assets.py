"""正式 GRPO 入口必须把 Shopping 专用 AgentLoop 和 Vanilla 设置接通。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GrpoRuntimeAssetsTest(unittest.TestCase):
    def test_agent_loop_config_loads_project_wrapper(self):
        config = (ROOT / "configs/verl/shop_agent_loops.yaml").read_text(encoding="utf-8")
        self.assertIn("shopping_tool_agent", config)
        self.assertIn("shopping_grpo.verl_adapter.agent_loop.ShoppingToolAgentLoop", config)

    def test_vanilla_config_uses_qwen_parser_and_environment_reward_only(self):
        config = (ROOT / "configs/verl/vanilla_grpo.yaml").read_text(encoding="utf-8")
        self.assertIn("adv_estimator: grpo", config)
        self.assertIn("format: qwen3_coder", config)
        self.assertIn("default_agent_loop: shopping_tool_agent", config)
        self.assertIn("use_remove_padding: false", config)
        self.assertIn("lora:\n      merge: true", config)
        self.assertIn(
            "worker_process_setup_hook: shopping_grpo.verl_compat.install_torch_padding_fallback",
            config,
        )
        self.assertIn("reward_model:\n  enable: false", config)
        self.assertNotIn("interaction_config_path", config)
        self.assertNotIn("prm", config.casefold())
        self.assertNotIn("lata", config.casefold())

    def test_server_launcher_uses_installed_verl_instead_of_reference_fork(self):
        launcher = ROOT / "scripts/run_vanilla_grpo.sh"
        self.assertTrue(launcher.is_file())
        content = launcher.read_text(encoding="utf-8")
        self.assertIn("verl.trainer.main_ppo", content)
        self.assertNotIn("agentic-grpo-longhorizon", content)
        self.assertNotIn("shop_interaction.json", content)

    def test_runtime_setup_applies_the_numpy_override(self):
        setup = (ROOT / "docs/grpo-runtime-setup.md").read_text(encoding="utf-8")
        self.assertIn("--override requirements-grpo-overrides.txt", setup)
        self.assertNotIn("uv pip check", setup)

    def test_grpo_dependencies_pin_the_supported_runtime(self):
        requirements = (ROOT / "requirements-grpo.txt").read_text(encoding="utf-8")
        self.assertIn("verl==0.8.0", requirements)
        self.assertIn("vllm==0.25.1", requirements)
        self.assertIn("transformers==5.11.0", requirements)
        self.assertIn("tensordict==0.10.0", requirements)
        self.assertIn("numpy==2.2.6", requirements)
        override = (ROOT / "requirements-grpo-overrides.txt").read_text(encoding="utf-8")
        self.assertEqual(override.strip(), "numpy==2.2.6")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
