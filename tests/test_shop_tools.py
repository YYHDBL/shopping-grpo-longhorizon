import unittest

from shopping_grpo.environment.tools import (
    SHOP_TOOL_SCHEMAS,
    tool_call_to_action,
)


class ShopToolsTest(unittest.TestCase):
    def test_search_products_maps_to_search_action(self):
        self.assertEqual(
            tool_call_to_action("search_products", {"query": "乳胶枕"}),
            "search[乳胶枕]",
        )

    def test_buy_now_maps_to_click_action(self):
        self.assertEqual(tool_call_to_action("buy_now", {}), "click[Buy Now]")

    def test_finish_without_purchase_maps_to_explicit_terminal_action(self):
        self.assertEqual(
            tool_call_to_action(
                "finish_without_purchase",
                {"reason": "no_suitable_product"},
            ),
            "finish[no_suitable_product]",
        )

    def test_tool_schemas_include_search_products(self):
        names = [schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS]

        self.assertIn("search_products", names)
        self.assertIn("finish_without_purchase", names)

    def test_tool_schemas_reject_undeclared_arguments(self):
        for schema in SHOP_TOOL_SCHEMAS:
            with self.subTest(tool=schema["function"]["name"]):
                self.assertFalse(schema["function"]["parameters"]["additionalProperties"])

    def test_tool_descriptions_state_current_page_constraints(self):
        schemas = {schema["function"]["name"]: schema["function"] for schema in SHOP_TOOL_SCHEMAS}

        self.assertIn("搜索功能是否可用: True", schemas["search_products"]["description"])
        self.assertIn("不得重复相同查询", schemas["search_products"]["description"])
        self.assertIn("最新 observation", schemas["open_product"]["description"])
        self.assertIn("不得选择导航按钮", schemas["select_option"]["description"])
        self.assertIn("影响可购买 variant 的必要规格轴", schemas["select_option"]["description"])
        self.assertIn("完整 variant 的实际价格", schemas["select_option"]["description"])
        self.assertIn("必须传 {}", schemas["view_description"]["description"])
        self.assertIn("Buy Now", schemas["buy_now"]["description"])
        self.assertIn("品类正确", schemas["buy_now"]["description"])
        self.assertIn("品牌、型号与核心功能、规格属性", schemas["buy_now"]["description"])
        self.assertNotIn("think", schemas)

    def test_finish_description_matches_reward_abstain_gate(self):
        schemas = {
            schema["function"]["name"]: schema["function"]
            for schema in SHOP_TOOL_SCHEMAS
        }

        description = schemas["finish_without_purchase"]["description"]
        self.assertIn("多次有实质差异的搜索", description)
        self.assertIn("多个候选核验", description)
        self.assertIn("没有可接受商品", description)
        self.assertIn("是否达到资格由环境判断", description)
        self.assertNotIn("至少 3 个", description)


if __name__ == "__main__":
    unittest.main()
