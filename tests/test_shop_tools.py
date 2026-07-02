import unittest

from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


class ShopToolsTest(unittest.TestCase):
    def test_search_products_maps_to_search_action(self):
        self.assertEqual(
            tool_call_to_action("search_products", {"query": "乳胶枕"}),
            "search[乳胶枕]",
        )

    def test_buy_now_maps_to_click_action(self):
        self.assertEqual(tool_call_to_action("buy_now", {}), "click[Buy Now]")

    def test_tool_schemas_include_search_products(self):
        names = [schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS]

        self.assertIn("search_products", names)


if __name__ == "__main__":
    unittest.main()
