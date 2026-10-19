import unittest
from copy import deepcopy
from types import SimpleNamespace

from shopping_grpo.training.grpo.adapter.legacy_reward import select_legacy_reward


class OriginalRewardTests(unittest.TestCase):
    def output(self):
        return SimpleNamespace(
            reward_score=0.0,
            extra_fields={
                "shopping": {
                    "done": True,
                    "infrastructure_invalid": False,
                    "reward_type": "reward_unverifiable",
                    "reward_valid": False,
                    "reward_unverifiable": True,
                    "reward": {"total": 0.0, "sampling_invalid": True, "reward_unverifiable": True},
                }
            },
        )

    def events(self, kind="gold_purchase", value=1.0, valid=True):
        return [
            {
                "result": {
                    "purchase": True,
                    "done": True,
                    "over": True,
                    "reward_detail": {
                        "evidence": {
                            "review_policy_version": "review6",
                            "legacy_result": {
                                "reward_type": kind,
                                "reward_valid": valid,
                                "reward": value,
                            },
                        }
                    },
                }
            }
        ]

    def test_legacy_success_over_review_unknown(self):
        o = self.output()
        events = self.events()
        original = deepcopy(events)
        select_legacy_reward(o, events)
        i = o.extra_fields["shopping"]
        self.assertEqual(o.reward_score, 1.0)
        self.assertEqual(i["reward"]["strict"], 1.0)
        self.assertFalse(i["reward"]["sampling_invalid"])
        self.assertTrue(i["reward_valid"])
        self.assertEqual(i["review6_diagnostics"]["reward_type"], "reward_unverifiable")
        self.assertEqual(events, original)

    def test_wrong_purchase_not_strict(self):
        o = self.output()
        select_legacy_reward(o, self.events("wrong_purchase", -1.0))
        self.assertEqual(o.reward_score, -1.0)
        self.assertEqual(o.extra_fields["shopping"]["reward"]["strict"], 0.0)

    def test_missing_legacy_fails_closed(self):
        o = self.output()
        e = self.events()
        e[0]["result"]["reward_detail"] = {}
        with self.assertRaises(RuntimeError):
            select_legacy_reward(o, e)

    def test_infrastructure_failure_not_rescued(self):
        o = self.output()
        o.extra_fields["shopping"]["infrastructure_invalid"] = True
        before = deepcopy(o.extra_fields)
        select_legacy_reward(o, self.events())
        self.assertEqual(o.extra_fields, before)

    def test_nonterminal_not_overridden(self):
        o = self.output()
        o.extra_fields["shopping"]["done"] = False
        before = deepcopy(o.extra_fields)
        select_legacy_reward(o, self.events())
        self.assertEqual(o.extra_fields, before)

    def test_invalid_reward_excluded(self):
        o = self.output()
        select_legacy_reward(o, self.events("reward_unverifiable", 0.0, False))
        self.assertTrue(o.extra_fields["shopping"]["reward"]["sampling_invalid"])

    def test_g_disabled_uses_original_terminal(self):
        events = self.events()
        detail = events[0]["result"]["reward_detail"]
        legacy = detail["evidence"]["legacy_result"]
        events[0]["result"]["reward_detail"] = {
            **legacy,
            "reward_version": "shopsimulator-reward-v3",
        }
        output = self.output()
        select_legacy_reward(output, events)
        self.assertEqual(output.reward_score, 1.0)
        self.assertNotIn("review6_diagnostics", output.extra_fields["shopping"])
