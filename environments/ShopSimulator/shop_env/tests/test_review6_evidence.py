import unittest
from unittest.mock import patch

from web_agent_site.engine import reviewed_reward as rr
from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE
from web_agent_site.engine.reviewed_public_constraints import product_checks


class EvidenceSemantics(unittest.TestCase):
    def checks(self, text, facts, option="汤锅"):
        with patch(
            "web_agent_site.engine.public_option_evidence.selected_option_evidence",
            return_value=[{"facts": facts}],
        ):
            return product_checks({}, text, {"颜色分类": option})

    def test_photo_alone_is_not_shipment(self):
        for facts in (
            {"included_components": ["车架", "汤锅", "配件"]},
            {"shipment_scope_explicit": True, "included_components": ["汤锅"]},
        ):
            with self.subTest(facts=facts):
                self.assertIn(
                    UNVERIFIABLE, [r["status"] for r in self.checks("需要小吃推车", facts)]
                )

    def test_explicit_bundle(self):
        self.assertEqual(
            self.checks(
                "需要小吃推车",
                {"shipment_scope_explicit": True, "included_components": ["车架", "汤锅", "配件"]},
            ),
            [],
        )

    def test_accessories_require_selected_shipping_proof(self):
        goal = {"expected_core_functions": ["附带配件"]}
        for facts, expected in [
            ({"included_components": ["配件"]}, UNVERIFIABLE),
            ({"included_components": ["配件"], "shipment_scope_explicit": True}, PASS),
        ]:
            with patch(
                "web_agent_site.engine.public_option_evidence.selected_option_evidence",
                return_value=[{"facts": facts}],
            ):
                r = rr._preferences({}, goal, {})["dimensions"]["core_functions"]["results"][0]
                self.assertEqual(r["status"], expected)
        with patch(
            "web_agent_site.engine.public_option_evidence.selected_option_evidence",
            return_value=[
                {"facts": {"included_components": ["配件"], "shipment_scope_explicit": True}}
            ],
        ):
            r = rr._preferences({"title": "不附带配件"}, goal, {})["dimensions"]["core_functions"][
                "results"
            ][0]
            self.assertEqual(r["status"], FAIL)

    def test_exclusive_protocol_overrides_unknown(self):
        r = self.checks("支持sata读取", {"exclusive_storage_protocols": ["nvme"]}, "单NVME")
        self.assertEqual(r[0]["status"], FAIL)

    def test_do_not_infer_protocol_exclusivity(self):
        for text, facts in [
            ("支持sata读取", {}),
            ("不支持sata读取", {"exclusive_storage_protocols": ["nvme"]}),
            ("无需支持sata读取", {"exclusive_storage_protocols": ["nvme"]}),
            ("不需要支持sata读取", {"exclusive_storage_protocols": ["nvme"]}),
            ("支持sata读取", {"exclusive_storage_protocols": ["sata", "nvme"]}),
        ]:
            with self.subTest(text=text, facts=facts):
                self.assertEqual(self.checks(text, facts, "盒子"), [])

    def test_type_c_cable_needs_inclusion_not_socket(self):
        self.assertEqual(
            self.checks(
                "配备Type-C数据线", {"included_cable_connectors": ["usb-a", "usb-c"]}, "盒子"
            ),
            [],
        )
        self.assertEqual(
            self.checks("配备Type-C数据线", {"ports": ["usb-c"]}, "盒子")[0]["status"], UNVERIFIABLE
        )
