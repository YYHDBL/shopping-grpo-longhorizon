import json
import unittest
from pathlib import Path


class ShopToolConfigTest(unittest.TestCase):
    def test_shop_tool_config_contains_search_tool(self):
        path = Path("configs/tool_config/shop_tools.yaml")
        config = json.loads(path.read_text())
        names = [tool["tool_schema"]["function"]["name"] for tool in config["tools"]]
        class_names = [tool["class_name"] for tool in config["tools"]]

        self.assertIn("search_products", names)
        self.assertIn(
            "shopping_grpo.verl_shop_tools.Shop_search_products_Tool",
            class_names,
        )


if __name__ == "__main__":
    unittest.main()
