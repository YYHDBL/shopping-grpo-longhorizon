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
        self.assertIn("reward_model:\n  enable: false", config)
        self.assertNotIn("prm", config.casefold())
        self.assertNotIn("lata", config.casefold())

    def test_server_launcher_exists(self):
        launcher = ROOT / "scripts/run_vanilla_grpo.sh"
        self.assertTrue(launcher.is_file())
        self.assertIn("verl.trainer.main_ppo", launcher.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
