import unittest

from web_agent_site.engine import reviewed_reward as rr
from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE
from web_agent_site.engine.reviewed_public_constraints import product_checks


class RemainingReview(unittest.TestCase):
    def gate(self, text, price, options=None):
        return rr.budget_gate({"status": PASS, "price": price}, {"instruction_text": text}, options)

    def test_exact_center_without_tolerance(self):
        for text, price in [
            ("价格大概1200元左右", 1200),
            ("价格在60元左右", 60),
            ("价格在2100左右", 2100),
            ("价格可以接受2万元左右", 20000),
            ("预算一千块左右", 1000),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.gate(text, price)["status"], PASS)
        self.assertEqual(self.gate("价格在60元左右", 61)["status"], UNVERIFIABLE)

    def test_sensitivity_not_truth(self):
        result = self.gate("预算20块左右", 22)
        self.assertEqual(result["status"], UNVERIFIABLE)
        self.assertTrue(
            result["evidence"]["approximation_diagnostics"][0]["sensitivity_only"]["0.1"]
        )
        self.assertFalse(
            result["evidence"]["approximation_diagnostics"][0]["sensitivity_only"]["0.05"]
        )

    def test_hard_budget_and_multiple_centers(self):
        self.assertEqual(self.gate("价格60元左右，但不超过50元", 60)["status"], FAIL)
        self.assertEqual(self.gate("价格60元左右，预算70元左右", 60)["status"], UNVERIFIABLE)
        self.assertEqual(self.gate("准备20元应该够了", 19)["status"], PASS)
        self.assertEqual(self.gate("准备20元应该够了", 21)["status"], FAIL)

    def test_payment_role(self):
        result = self.gate("想预定，定金可以接受10000元", 10000, {"颜色分类": "标准机定金"})
        self.assertEqual(result["status"], PASS)
        self.assertEqual(
            self.gate("想预定，定金可以接受10000元", 10000, {"颜色分类": "标准机全款"})["status"],
            UNVERIFIABLE,
        )
        result = self.gate("定金150元，全款730元左右", 270, {"颜色分类": "定金150元全款731元"})
        self.assertEqual(result["status"], UNVERIFIABLE)
        self.assertEqual(result["evidence"]["payment"]["status"], "data_conflict")
        self.assertEqual(
            self.gate("定金150元，全款730元左右", 150, {"颜色分类": "定金150元全款731元"})[
                "status"
            ],
            UNVERIFIABLE,
        )

    def test_unrequested_code(self):
        from web_agent_site.engine.reward_features import compile_reward_features

        p = {
            "asin": "123456789012",
            "category": "灯",
            "customization_options": {
                "颜色分类": [
                    {"value": "42寸 白色 31A款", "price": 345},
                    {"value": "42寸 白色 33C款", "price": 410},
                    {"value": "42寸 黑色 33C款", "price": 410},
                ]
            },
        }
        for text, choice, status in [
            ("要42寸白色", "42寸 白色 33C款", PASS),
            ("要31A款", "42寸 白色 33C款", FAIL),
            ("要42寸白色", "42寸 黑色 33C款", FAIL),
        ]:
            goal = {
                "instruction_text": text,
                **compile_reward_features(
                    {
                        "instruction": text,
                        "attributes": [],
                        "instruction_options": ["42寸 白色 31A款"],
                    },
                    p,
                ),
            }
            self.assertEqual(
                rr._preferences(p, goal, {"颜色分类": choice})["dimensions"]["key_options"][
                    "results"
                ][0]["status"],
                status,
            )

    def test_alias_preserves_negative(self):
        for claim, word in [("防结露", "防凝露"), ("官方正品", "官网正品")]:
            for prefix, status in [("", PASS), ("不支持", FAIL)]:
                result = rr._preferences(
                    {"title": prefix + word, "attribute": [prefix + word]},
                    {"expected_core_functions": [claim]},
                    {},
                )
                self.assertEqual(
                    result["dimensions"]["core_functions"]["results"][0]["status"], status
                )

    def test_missing_claim_and_bundle(self):
        self.assertEqual(
            product_checks({"title": "包装破损必赔"}, "包装不容易破", {})[0]["status"], UNVERIFIABLE
        )
        self.assertEqual(
            product_checks({"title": "小吃推车汤锅"}, "需要小吃推车", {"颜色分类": "汤锅"})[0][
                "status"
            ],
            UNVERIFIABLE,
        )
        self.assertEqual(
            product_checks(
                {"title": "美牙仪套装"}, "完整一套美牙仪", {"颜色分类": "官方原装美白凝聚"}
            )[0]["status"],
            UNVERIFIABLE,
        )

    def test_no_gram_to_ml_or_color_code_inference(self):
        self.assertEqual(
            product_checks({}, "净含量300毫升", {"净含量": "300g"})[0]["status"], UNVERIFIABLE
        )
        self.assertEqual(
            product_checks({}, "要深蓝色", {"颜色分类": "100074690"})[0]["status"], UNVERIFIABLE
        )
