import unittest

from web_agent_site.engine import reviewed_reward as rr
from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE


class ReviewedRewardTest(unittest.TestCase):
    def gate(self, instruction, price):
        return rr.budget_gate({"status": PASS, "price": price}, {"instruction_text": instruction})[
            "status"
        ]

    def test_explicit_range_has_both_ends(self):
        self.assertEqual(self.gate("价格在800–820元", 799), FAIL)
        self.assertEqual(self.gate("价格在800–820元", 810), PASS)
        self.assertEqual(self.gate("价格在800–820元", 821), FAIL)

    def test_approximate_is_not_invented_tolerance(self):
        self.assertEqual(self.gate("预算在20块左右", 22), UNVERIFIABLE)
        self.assertEqual(self.gate("预算在20元左右", 100), UNVERIFIABLE)

    def test_lower_only_not_upper(self):
        self.assertEqual(self.gate("价格在1300元以上", 1400), PASS)
        self.assertEqual(self.gate("价格在1300元以上", 1200), FAIL)

    def test_explicit_conflict_even_with_uncertainty(self):
        self.assertEqual(self.gate("预算20元左右，但不超过25元", 30), FAIL)

    def test_missing_price_and_conflicting_bounds(self):
        self.assertEqual(self.gate("不超过30元", None), UNVERIFIABLE)
        self.assertEqual(self.gate("至少40元，不超过30元", 35), UNVERIFIABLE)

    def test_no_budget_is_not_hidden_budget(self):
        self.assertEqual(self.gate("要红色耳塞", 50), PASS)

    def test_unsupported_budget_is_unknown(self):
        self.assertEqual(self.gate("预算三百三十元", 294), PASS)

    def test_yuan_in_element_is_not_money(self):
        self.assertEqual(self.gate("要包含熊耳朵的元素，价格在60元以内", 50), PASS)
        self.assertEqual(self.gate("奥特曼元素的鞋", 350), PASS)

    def test_plain_budget_amount(self):
        self.assertEqual(self.gate("预算35元", 36), FAIL)

    def test_missing_function_is_unknown_but_negation_fails(self):
        goal = {"expected_core_functions": ["休眠"]}
        d = rr._preferences({"title": "鼠标", "attribute": ["无线"]}, goal, {})["dimensions"][
            "core_functions"
        ]["results"][0]
        self.assertEqual(d["status"], UNVERIFIABLE)
        d = rr._preferences({"title": "鼠标", "attribute": ["不支持休眠"]}, goal, {})["dimensions"][
            "core_functions"
        ]["results"][0]
        self.assertEqual(d["status"], FAIL)

    def test_narrow_alias_preserves_negation(self):
        goal = {"expected_core_functions": ["推车架子"]}
        for text, status in [("折叠推车架", PASS), ("不含推车架", FAIL)]:
            d = rr._preferences({"title": text, "attribute": [text]}, goal, {})["dimensions"][
                "core_functions"
            ]["results"][0]
            self.assertEqual(d["status"], status)

    def test_unrequested_installation_only(self):
        from web_agent_site.engine.reward_features import compile_reward_features

        product = {
            "asin": "123456789012",
            "category": "家居›窗帘",
            "customization_options": {
                "尺寸": [
                    {"value": "宽3米打孔", "price": 30},
                    {"value": "宽3米挂钩", "price": 30},
                    {"value": "宽2米挂钩", "price": 20},
                ]
            },
        }
        for instruction, choice, status in [
            ("要宽3米窗帘", "宽3米挂钩", PASS),
            ("要宽3米打孔窗帘", "宽3米挂钩", FAIL),
            ("要宽3米窗帘", "宽2米挂钩", FAIL),
        ]:
            goal = {
                "instruction_text": instruction,
                **compile_reward_features(
                    {
                        "instruction": instruction,
                        "attributes": [],
                        "instruction_options": ["宽3米打孔"],
                    },
                    product,
                ),
            }
            result = rr._preferences(product, goal, {"尺寸": choice})["dimensions"]["key_options"][
                "results"
            ][0]
            self.assertEqual(result["status"], status)


if __name__ == "__main__":
    unittest.main()
