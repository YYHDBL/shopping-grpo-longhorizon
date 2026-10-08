import unittest

from web_agent_site.engine import reviewed_reward as rr
from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE
from web_agent_site.engine.reviewed_public_constraints import selected_checks


class PublicChecks(unittest.TestCase):
    def gate(self, text, price):
        return rr.budget_gate({"status": PASS, "price": price}, {"instruction_text": text})[
            "status"
        ]

    def test_chinese_money(self):
        for text, price, status in [
            ("预算在三百三十元", 330, PASS),
            ("预算在三百三十元", 331, FAIL),
            ("不超过两千元", 2001, FAIL),
            ("至少一百零五元", 104, FAIL),
            ("预算三百五元", 350, UNVERIFIABLE),
            ("预算五块钱左右", 5, PASS),
        ]:
            with self.subTest(text=text, price=price):
                self.assertEqual(self.gate(text, price), status)

    def test_count_units_not_currency(self):
        self.assertEqual(self.gate("能装两块2.5英寸磁盘", 999), PASS)
        self.assertEqual(self.gate("两块硬盘，预算不超过500元", 501), FAIL)

    def test_new_explicit_wording(self):
        self.assertEqual(self.gate("价格不要超过80元", 81), FAIL)
        self.assertEqual(self.gate("价格控制在35元", 35), PASS)
        self.assertEqual(self.gate("价格控制在35元", 36), FAIL)

    def test_mixed_approximate_and_hard(self):
        self.assertEqual(self.gate("预算20元左右但不超过25元", 26), FAIL)
        self.assertEqual(self.gate("预算20元左右但不超过25元", 24), UNVERIFIABLE)

    def test_selected_conflicts(self):
        for text, chosen in [
            ("尺寸小于5寸", "6寸滚筒"),
            ("小型低音炮", "sub大低音炮"),
            ("闲置时能自动进入低功耗", "不休眠款"),
        ]:
            with self.subTest(text=text):
                self.assertIn(
                    FAIL, [x["status"] for x in selected_checks(text, {"颜色分类": chosen})]
                )

    def test_color_unknown_and_not_other_options(self):
        x = selected_checks("颜色为白色", {"颜色分类": "奶油色"})
        self.assertEqual(x[0]["status"], UNVERIFIABLE)

    def test_component_not_assumed_assembly(self):
        self.assertEqual(
            selected_checks("一款适用威兰达的空调旋钮", {"颜色分类": "温度传感器"})[0]["status"],
            UNVERIFIABLE,
        )

    def test_bounds_and_ambiguity(self):
        self.assertEqual(selected_checks("尺寸小于5寸", {"尺寸": "5寸"})[0]["status"], FAIL)
        self.assertEqual(selected_checks("尺寸小于5寸", {"尺寸": "4寸"})[0]["status"], PASS)
        self.assertEqual(
            selected_checks("尺寸小于5寸", {"尺寸": "4寸/6寸"})[0]["status"], UNVERIFIABLE
        )

    def test_literal_color_morphemes(self):
        self.assertEqual(selected_checks("颜色要蓝色", {"颜色分类": "海军蓝"}), [])
        self.assertEqual(selected_checks("颜色要黄色", {"颜色分类": "黄爱心"}), [])
        self.assertEqual(
            selected_checks("颜色为白色", {"颜色分类": "100074689"})[0]["status"], UNVERIFIABLE
        )

    def test_mass_units_and_ambiguous_selected(self):
        self.assertEqual(
            selected_checks("规格为1.5kg", {"颜色分类": "一斤 样品"})[0]["status"], FAIL
        )
        self.assertEqual(selected_checks("规格为1.5kg", {"颜色分类": "三斤"})[0]["status"], PASS)
        self.assertEqual(
            selected_checks("规格为1.5kg", {"颜色分类": "0.5kg/1.5kg"})[0]["status"], UNVERIFIABLE
        )
        self.assertEqual(
            selected_checks("规格为1.5kg左右", {"颜色分类": "一斤"})[0]["status"], UNVERIFIABLE
        )
