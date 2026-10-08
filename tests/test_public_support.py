import json
import unittest
from copy import deepcopy

from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.public_support import (
    available_schemas,
    purchase_check,
    resolve_option_call,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.evaluation.rollout import collect_for_task


def page(selected=None, price="19", options=None):
    options = options or {"颜色": ["自然黑 1瓶 【500ml】"], "容量": ["单瓶"]}
    return "\n".join(
        [
            "[SHOPPING_OBSERVATION_V2]",
            "page_type: product_detail",
            "asin: 123456789012",
            "price: " + price,
            "selected_options: " + json.dumps(selected or {}, ensure_ascii=False),
            "available_options: " + json.dumps(options, ensure_ascii=False),
            "搜索功能是否可用: False",
            "可点击的按钮: "
            + json.dumps(
                ["buy now", "back to search", "< prev", "description", "features"]
                + [v for vs in options.values() for v in vs],
                ensure_ascii=False,
            ),
        ]
    )


def call(name, args=None):
    return {
        "id": "call",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
    }


class PublicSupportTest(unittest.TestCase):
    def test_label_schema_preserves_exact_whitespace_and_shared_tools(self):
        before = deepcopy(SHOP_TOOL_SCHEMAS)
        obs = page(options={"颜色": ["军绿色", "桔色", "a  b"]})
        schema = next(
            s
            for s in available_schemas(SHOP_TOOL_SCHEMAS, obs, option_labels=True)
            if s["function"]["name"] == "select_option"
        )
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["value"]["enum"],
            ["军绿色", "桔色", "a  b"],
        )
        self.assertNotIn("opt:", schema["function"]["description"])
        self.assertEqual(SHOP_TOOL_SCHEMAS, before)

    def test_current_page_ids_resolve_to_exact_button(self):
        c, event = resolve_option_call(
            call("select_option", {"value": "opt:123456789012:1"}), page()
        )
        self.assertIsNotNone(event)
        self.assertIn(
            json.loads(c["function"]["arguments"])["value"], ["自然黑 1瓶 【500ml】", "单瓶"]
        )
        c, event = resolve_option_call(
            call("select_option", {"value": "opt:999999999999:1"}), page()
        )
        self.assertIsNone(event)

    def test_only_unique_whitespace_normalization_no_semantic_guess(self):
        c, _event = resolve_option_call(
            call("select_option", {"value": "自然黑 1瓶【500ml】"}), page()
        )
        self.assertEqual(json.loads(c["function"]["arguments"])["value"], "自然黑 1瓶 【500ml】")
        for options, value in [
            ({"轴": ["a b", "ab"]}, "a  b"),
            ({"轴": ["材料包含工具"]}, "材料包包含工具"),
            ({"轴": ["不休眠"]}, "休眠"),
        ]:
            self.assertIsNone(
                resolve_option_call(call("select_option", {"value": value}), page(options=options))[
                    1
                ]
            )

    def test_search_page_cannot_select_or_view_product(self):
        obs = 'page_type: search_results\n1|123456789012|10|x\n搜索功能是否可用: False\n可点击的按钮: ["123456789012", "back to search"]'
        schemas = available_schemas(SHOP_TOOL_SCHEMAS, obs)
        self.assertEqual(
            {s["function"]["name"] for s in schemas},
            {"open_product", "back_to_search", "finish_without_purchase"},
        )
        self.assertIsNone(
            resolve_option_call(call("select_option", {"value": "opt:123456789012:1"}), obs)[1]
        )

    def test_no_nonexistent_attributes_and_no_shared_schema_mutation(self):
        before = deepcopy(SHOP_TOOL_SCHEMAS)
        schemas = available_schemas(SHOP_TOOL_SCHEMAS, page())
        self.assertNotIn("view_attributes", [s["function"]["name"] for s in schemas])
        self.assertEqual(before, SHOP_TOOL_SCHEMAS)

    def test_missing_singleton_blocks_before_budget(self):
        r = purchase_check("不超过20元", page({"颜色": "自然黑 1瓶 【500ml】"}))
        self.assertEqual(r["reason"], "variant_axes_unselected")
        self.assertIn("容量", r["feedback"])
        self.assertIsNone(
            purchase_check("不超过20元", page({"颜色": "自然黑 1瓶 【500ml】", "容量": "单瓶"}))[
                "reason"
            ]
        )

    def test_budget_exact_bounds_and_approximation(self):
        sel = {"颜色": "自然黑 1瓶 【500ml】", "容量": "单瓶"}
        for text in ["不超20元", "价格控制在20元", "20能搞定", "价格在20-30元之间"]:
            self.assertIsNotNone(purchase_check(text, page(sel, "40"))["reason"])
        self.assertIsNone(purchase_check("约20元左右", page(sel, "22"))["reason"])
        self.assertIsNone(purchase_check("重量不超过5斤", page(sel, "40"))["reason"])
        self.assertEqual(
            purchase_check("约20元", page(sel, "19 to 22"))["reason"], "variant_price_unknown"
        )

    def test_public_renderer_does_not_emit_nested_hidden_fields(self):
        state = {
            "observation_version": "shopping-observation-v2",
            "page_type": "product_detail",
            "actions": ["buy now"],
            "product": {"asin": "123456789012", "title": "杯子"},
            "public_details": {"description": "可见描述", "goal": "SECRET", "answer": "SECRET"},
        }
        text = render_structured_observation(state)
        self.assertIn("可见描述", text)
        self.assertNotIn("SECRET", text)

    def test_execution_records_resolution_and_requires_remaining_axis(self):
        class Env:
            def __init__(self):
                self.selected = {}
                self.actions = []

            def reset(self, tid):
                return {"instruction": "Instruction: 20元以内\n" + page()}

            def step(self, action):
                self.actions.append(action)
                if action == "click[自然黑 1瓶 【500ml】]":
                    self.selected["颜色"] = "自然黑 1瓶 【500ml】"
                elif action == "click[单瓶]":
                    self.selected["容量"] = "单瓶"
                elif action == "click[Buy Now]":
                    return {"instruction": "done", "done": True, "over": True, "reward": 1}
                else:
                    raise AssertionError(action)
                return {"instruction": page(self.selected), "done": False, "reward": 0}

            def release(self):
                pass

        class Client:
            def __init__(self):
                self.calls = iter(
                    [
                        call("buy_now"),
                        call("select_option", {"value": "自然黑 1瓶【500ml】"}),
                        call("select_option", {"value": "单瓶"}),
                        call("buy_now"),
                    ]
                )

            def complete(self, messages, tools):
                return {"role": "assistant", "content": None, "tool_calls": [next(self.calls)]}

        env = Env()
        r = collect_for_task(
            {"task_id": 1}, Client(), env_factory=lambda **_: env, public_support=True
        )
        self.assertEqual(r["status"], "done")
        self.assertEqual(len(r["blocked_tool_calls"]), 1)
        self.assertEqual(
            env.actions, ["click[自然黑 1瓶 【500ml】]", "click[单瓶]", "click[Buy Now]"]
        )
        self.assertTrue(
            any(e.get("method") == "unique_whitespace_match" for e in r["public_support_events"])
        )


if __name__ == "__main__":
    unittest.main()
