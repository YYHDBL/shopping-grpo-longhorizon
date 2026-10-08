import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web_agent_site.engine.observation import build_observation_state
from web_agent_site.engine.public_option_evidence import option_evidence
from web_agent_site.engine.reviewed_public_constraints import product_checks


class OptionEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "evidence.json"
        self.record = {
            "asin": "p",
            "axis": "颜色分类",
            "option_value": "奶油色",
            "source_url": "https://example.org/a.jpg",
            "caption": "图片标注遮光度95%",
            "facts": {"blackout_percent": 95},
        }
        self.path.write_text(
            json.dumps({"version": "public-option-evidence-v1", "records": [self.record]})
        )
        self.product = {
            "asin": "p",
            "title": "全遮光窗帘",
            "Description": "全遮光窗帘",
            "options": {"颜色分类": ["奶油色", "白色"]},
            "customization_options": {
                "颜色分类": [
                    {"value": "奶油色", "image": "https://example.org/a.jpg"},
                    {"value": "白色", "image": "https://example.org/b.jpg"},
                ]
            },
            "instructions": ["hidden should never render"],
        }

    def test_off_by_default(self):
        with patch.dict(os.environ, {"SHOP_PUBLIC_OPTION_EVIDENCE": ""}):
            self.assertEqual(option_evidence(self.product), [])

    def test_bound_selected_conflict_not_other_option(self):
        with patch.dict(os.environ, {"SHOP_PUBLIC_OPTION_EVIDENCE": str(self.path)}):
            checks = product_checks(self.product, "要100%遮光", {"颜色分类": "奶油色"})
            self.assertEqual(checks[0]["status"], "fail")
            self.assertEqual(product_checks(self.product, "要100%遮光", {"颜色分类": "白色"}), [])

    def test_reject_stale_binding(self):
        self.product["customization_options"]["颜色分类"][0]["image"] = (
            "https://example.org/changed.jpg"
        )
        with (
            patch.dict(os.environ, {"SHOP_PUBLIC_OPTION_EVIDENCE": str(self.path)}),
            self.assertRaises(ValueError),
        ):
            option_evidence(self.product)

    def test_public_state_includes_source_not_hidden_answer(self):
        with patch.dict(
            os.environ,
            {"SHOP_PUBLIC_OPTION_EVIDENCE": str(self.path), "SHOPSIM_PUBLIC_SUPPORT": "1"},
        ):
            state = build_observation_state(
                page_type="product_detail",
                session={"asin": "p", "options": {"颜色分类": "奶油色"}},
                product_item_dict={"p": self.product},
                available_actions={"clickables": []},
            )
            self.assertEqual(
                state["public_details"]["option_evidence"][0]["facts"]["blackout_percent"], 95
            )
            self.assertNotIn("hidden should never render", json.dumps(state))

    def test_same_urls_wrong_product_not_used(self):
        self.product["asin"] = "another"
        with patch.dict(os.environ, {"SHOP_PUBLIC_OPTION_EVIDENCE": str(self.path)}):
            self.assertEqual(option_evidence(self.product), [])
