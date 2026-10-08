import json
import unittest

from shopping_grpo.environment.budget import check_purchase_budget, extract_budget
from shopping_grpo.evaluation.rollout import collect_for_task


def page(price):
    return f'page_type: product_detail\nprice: {price}\n可点击的按钮: ["Buy Now", "cheap", "Back to Search"]'


def call(name, arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": name,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {})},
            }
        ],
    }


class BudgetParserTest(unittest.TestCase):
    def test_upper_bound_and_inclusive_boundary(self):
        for instruction in [
            "250元以内",
            "价格不超过250元",
            "预算在250元以下",
            "价格不能超过250块钱",
        ]:
            with self.subTest(instruction=instruction):
                self.assertEqual(
                    check_purchase_budget(instruction, page("299"))["reason"],
                    "budget_above_maximum",
                )
                self.assertIsNone(check_purchase_budget(instruction, page("250.00"))["reason"])

    def test_interval_both_ends(self):
        for instruction in ["价格在800-820元之间", "预算800至820元"]:
            with self.subTest(instruction=instruction):
                self.assertEqual(
                    check_purchase_budget(instruction, page("799"))["reason"],
                    "budget_below_minimum",
                )
                self.assertEqual(
                    check_purchase_budget(instruction, page("850"))["reason"],
                    "budget_above_maximum",
                )
                for price in ["800", "810", "820"]:
                    self.assertIsNone(check_purchase_budget(instruction, page(price))["reason"])

    def test_vague_price_not_silently_hard_capped(self):
        for instruction in [
            "价格250元左右",
            "预算差不多七百块",
            "价格最好在250元以下",
            "预算有70元",
        ]:
            self.assertFalse(extract_budget(instruction)["covered"])

    def test_non_price_numbers_not_used(self):
        self.assertFalse(extract_budget("重量不超过5斤，质保至少1年，尺寸在800-820毫米")["covered"])

    def test_unknown_or_range_price_does_not_pass(self):
        for price in ["199 to 299", "unknown", "nan", "-3"]:
            self.assertEqual(
                check_purchase_budget("250元以内", page(price))["reason"], "budget_price_unverified"
            )

    def test_current_price_not_historical_or_embedded(self):
        self.assertEqual(
            check_purchase_budget("250元以内", page("299") + "\nold_price: 199")["reason"],
            "budget_above_maximum",
        )
        self.assertEqual(
            check_purchase_budget("250元以内", "description: price: 199")["reason"],
            "budget_price_unverified",
        )

    def test_decimal_and_lower_only(self):
        self.assertIsNone(check_purchase_budget("不超过250.5元", page("250.50"))["reason"])
        self.assertEqual(
            check_purchase_budget("不低于800元", page("799.99"))["reason"], "budget_below_minimum"
        )

    def test_no_budget_no_intervention_even_with_unknown_price(self):
        self.assertEqual(
            check_purchase_budget("想找个黑色的", page("unknown"))["status"], "uncovered"
        )


class Env:
    def __init__(self):
        self.actions = []
        self.released = False
        self.price = "299"

    def reset(self, task_id):
        return {"instruction": "Instruction: 找250元以内的商品\n" + page(self.price)}

    def step(self, action):
        self.actions.append(action)
        if action == "click[cheap]":
            self.price = "249"
            return {"instruction": page(self.price), "reward": 0.0, "done": False}
        if action == "click[Buy Now]":
            return {
                "instruction": "done",
                "reward": 1.0,
                "done": True,
                "over": True,
                "purchase": {"price": float(self.price)},
            }
        raise AssertionError(action)

    def release(self):
        self.released = True


class Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.snapshots = []

    def complete(self, messages, tools):
        self.snapshots.append(json.loads(json.dumps(messages)))
        return self.replies.pop(0)


class BudgetExecutionTest(unittest.TestCase):
    def test_blocks_before_purchase_and_allows_recovery(self):
        env = Env()
        client = Client(
            [call("buy_now"), call("select_option", {"value": "cheap"}), call("buy_now")]
        )
        result = collect_for_task(
            {"task_id": 1}, client, env_factory=lambda **_: env, budget_guard=True
        )
        self.assertEqual(env.actions, ["click[cheap]", "click[Buy Now]"])
        self.assertEqual(result["terminal_result"]["purchase"]["price"], 249)
        self.assertEqual([x["status"] for x in result["budget_checks"]], ["blocked", "allowed"])
        self.assertIn("购买未执行", client.snapshots[1][-1]["content"])
        self.assertIn("299", client.snapshots[1][-1]["content"])
        self.assertEqual(len(result["blocked_tool_calls"]), 1)
        self.assertTrue(env.released)
        self.assertIsNone(result["error"])

    def test_disabled_preserves_original_purchase(self):
        env = Env()
        client = Client([call("buy_now")])
        result = collect_for_task({"task_id": 1}, client, env_factory=lambda **_: env)
        self.assertEqual(env.actions, ["click[Buy Now]"])
        self.assertEqual(result["terminal_result"]["purchase"]["price"], 299)
        self.assertEqual(result["budget_checks"], [])

    def test_repeated_rejected_purchase_never_reaches_environment(self):
        env = Env()
        client = Client([call("buy_now")] * 3)
        result = collect_for_task(
            {"task_id": 1}, client, env_factory=lambda **_: env, budget_guard=True
        )
        self.assertEqual(env.actions, [])
        self.assertEqual(result["status"], "invalid_action_limit")
        self.assertEqual(len(result["budget_checks"]), 3)
        self.assertTrue(env.released)


if __name__ == "__main__":
    unittest.main()
