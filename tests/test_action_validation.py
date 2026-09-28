import unittest

from shopping_grpo.environment.actions import (
    action_reject_reason,
    product_ids,
    subpage_visit_key,
)


class ActionValidationTest(unittest.TestCase):
    def test_select_option_rejects_navigation_button(self):
        """规格工具不能把页面导航按钮伪装成一个规格值。"""
        observation = '商品页\n\n可点击的按钮: ["< Prev", "糖果粉"]'

        reason = action_reject_reason("select_option", {"value": "< Prev"}, observation)

        self.assertEqual(reason, "select_option_is_navigation_button")

    def test_select_option_allows_current_product_option(self):
        observation = '商品页\n\n可点击的按钮: ["< Prev", "糖果粉"]'

        self.assertIsNone(action_reject_reason("select_option", {"value": "糖果粉"}, observation))

    def test_post_selection_allows_current_page_navigation(self):
        """规格选择不改变环境的可用动作；只校验当前页面是否真的有按钮。"""
        observation = '商品页\n\n可点击的按钮: ["Description", "Buy Now"]'

        reason = action_reject_reason("view_description", {}, observation)

        self.assertIsNone(reason)

    def test_rejects_schema_extra_argument_before_executing_tool(self):
        """无参数工具携带垃圾字段时，不能静默丢弃字段后继续执行。"""
        observation = '商品页\n\n可点击的按钮: ["Buy Now"]'

        reason = action_reject_reason("buy_now", {"string": "true"}, observation)

        self.assertEqual(reason, "schema_extra_arguments:string")

    def test_rejects_missing_required_argument_before_executing_tool(self):
        observation = "搜索功能是否可用: True"

        reason = action_reject_reason("search_products", {}, observation)

        self.assertEqual(reason, "schema_missing_arguments:query")

    def test_structured_observation_accepts_real_eleven_digit_product_id(self):
        asin = "35842622441"
        observation = (
            "[SHOPPING_OBSERVATION_V2]\n"
            "1|35842622441|158.0|泰国乳胶枕\n"
            '可点击的按钮: ["back to search", "35842622441"]'
        )
        self.assertEqual(product_ids(observation), [asin])
        self.assertIsNone(
            action_reject_reason("open_product", {"asin": asin}, observation)
        )

    def test_unrelated_long_number_is_not_treated_as_product(self):
        observation = (
            "商品描述包含电话号码 13800138000 和价格 12345678"
            '\n\n可点击的按钮: ["back to search"]'
        )
        self.assertEqual(product_ids(observation), [])

    def test_subpage_repeated_visit_is_rejected(self):
        """同一商品同一子页第二次访问被守卫拒绝。"""
        observation = (
            "[SHOPPING_OBSERVATION_V2]\n"
            "page_type: product_detail\n"
            "asin: 747848614498\n"
            "title: 测试商品\n"
            '可点击的按钮: ["back to search", "description", "buy now"]'
        )
        self.assertIsNone(
            action_reject_reason("view_description", {}, observation)
        )
        self.assertIsNone(
            action_reject_reason(
                "view_description",
                {},
                observation,
                visited_subpages={("747848614498", "Features")},
            )
        )
        self.assertEqual(
            action_reject_reason(
                "view_description",
                {},
                observation,
                visited_subpages={("747848614498", "Description")},
            ),
            "subpage_already_visited",
        )

    def test_subpage_key_extracts_asin_from_detail_page(self):
        observation = "asin: 747848614498\ntitle: 测试商品\n"
        self.assertEqual(
            subpage_visit_key("view_reviews", observation),
            ("747848614498", "Reviews"),
        )
        self.assertIsNone(subpage_visit_key("buy_now", observation))


if __name__ == "__main__":
    unittest.main()
