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

    def test_tool_schemas_reject_undeclared_arguments(self):
        for schema in SHOP_TOOL_SCHEMAS:
            with self.subTest(tool=schema["function"]["name"]):
                self.assertFalse(schema["function"]["parameters"]["additionalProperties"])

    def test_tool_descriptions_state_current_page_constraints(self):
        schemas = {schema["function"]["name"]: schema["function"] for schema in SHOP_TOOL_SCHEMAS}

        self.assertIn("搜索功能是否可用: True", schemas["search_products"]["description"])
        self.assertIn("最新 observation", schemas["open_product"]["description"])
        self.assertIn("不得选择导航按钮", schemas["select_option"]["description"])
        self.assertIn("必须传 {}", schemas["view_description"]["description"])
        self.assertIn("Buy Now", schemas["buy_now"]["description"])


if __name__ == "__main__":
    unittest.main()
