import json
import unittest
from pathlib import Path


class Stage4ConfigsTest(unittest.TestCase):
    def test_interaction_config_points_to_shop_interaction(self):
        config = json.loads(Path("configs/interaction_config/shop.yaml").read_text())
        item = config["interaction"][0]

        self.assertEqual(item["name"], "shop")
        self.assertEqual(
            item["class_name"],
            "shopping_grpo.verl_shop_interaction.ShopInteraction",
        )

    def test_tiny_grpo_config_uses_shop_tools_and_group_size_two(self):
        config = json.loads(Path("configs/train/grpo/shop_tiny_grpo.yaml").read_text())

        self.assertEqual(config["data"]["tool_config_path"], "configs/tool_config/shop_tools.yaml")
        self.assertEqual(
            config["actor_rollout_ref"]["rollout"]["multi_turn"]["interaction_config_path"],
            "configs/interaction_config/shop.yaml",
        )
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
