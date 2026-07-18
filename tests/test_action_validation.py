import unittest

from shopping_grpo.action_validation import action_reject_reason


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


if __name__ == "__main__":
    unittest.main()
