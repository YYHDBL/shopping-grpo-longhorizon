import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.sft_data import (
    acceptance_reasons,
    build_sft_row,
    process_raw_trajectories,
)


def assistant_tool(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


def tool_message(call_id, name, content):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def accepted_trajectory():
    asin = "100000000001"
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "帮我买乳胶枕"},
        assistant_tool("search_products", {"query": "乳胶枕"}, "call_1"),
        tool_message("call_1", "search_products", f"results [SEP] {asin} [SEP] 乳胶枕"),
        assistant_tool("open_product", {"asin": asin}, "call_2"),
        tool_message(
            "call_2",
            "open_product",
            'detail\n\n可点击的按钮: ["满天星", "Buy Now"]',
        ),
        assistant_tool("select_option", {"value": "满天星"}, "call_3"),
        tool_message("call_3", "select_option", 'selected\n\n可点击的按钮: ["Buy Now"]'),
        assistant_tool("buy_now", {}, "call_4"),
        tool_message("call_4", "buy_now", "done"),
    ]
    return {
        "trajectory_id": "traj-ok",
        "task_id": 0,
        "status": "done",
        "done": True,
        "error": None,
        "messages": messages,
        "steps": [
            {
                "tool_name": "search_products",
                "tool_call": messages[2]["tool_calls"][0],
                "env_action": "search[乳胶枕]",
                "done": False,
            },
            {
                "tool_name": "open_product",
                "tool_call": messages[4]["tool_calls"][0],
                "env_action": f"click[{asin}]",
                "done": False,
            },
            {
                "tool_name": "select_option",
                "tool_call": messages[6]["tool_calls"][0],
                "env_action": "click[满天星]",
                "done": False,
            },
            {
                "tool_name": "buy_now",
                "tool_call": messages[8]["tool_calls"][0],
                "env_action": "click[Buy Now]",
                "done": True,
            },
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": True},
            "goal": {"asin": asin},
            "purchase": {"asin": asin},
        },
    }


class SftDataTest(unittest.TestCase):
    def test_acceptance_reasons_accepts_successful_rule_checked_trajectory(self):
        accepted, reasons = acceptance_reasons(accepted_trajectory())

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_acceptance_reasons_rejects_failed_reward_components(self):
        traj = accepted_trajectory()
        traj["terminal_result"]["reward_detail"]["r_option"] = 0

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("reward_detail.r_option_not_1", reasons)

    def test_acceptance_reasons_rejects_missing_buy(self):
        traj = accepted_trajectory()
        traj["steps"] = traj["steps"][:-1]

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("missing_buy", reasons)

    def test_acceptance_reasons_rejects_parallel_tool_calls(self):
        traj = accepted_trajectory()
        traj["messages"][2]["tool_calls"].append(
            {
                "id": "call_extra",
                "type": "function",
                "function": {"name": "open_product", "arguments": '{"asin":"A2"}'},
            }
        )

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("message_2.multiple_tool_calls", reasons)

    def test_acceptance_reasons_rejects_product_not_on_latest_page(self):
        traj = accepted_trajectory()
        stale_asin = "999999999999"
        traj["steps"][1]["tool_call"]["function"]["arguments"] = json.dumps({"asin": stale_asin})
        traj["steps"][1]["env_action"] = f"click[{stale_asin}]"

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("step_1.click_not_in_previous_observation", reasons)

    def test_acceptance_reasons_rejects_option_not_on_latest_page(self):
        traj = accepted_trajectory()
        traj["messages"][5]["content"] = 'detail\n\n可点击的按钮: ["别的规格", "Buy Now"]'

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("step_2.click_not_in_previous_observation", reasons)

    def test_acceptance_reasons_rejects_buy_not_on_latest_page(self):
        traj = accepted_trajectory()
        traj["messages"][7]["content"] = 'selected\n\n可点击的按钮: ["满天星"]'

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("step_3.click_not_in_previous_observation", reasons)

    def test_acceptance_reasons_rejects_navigation_after_option_selection(self):
        traj = accepted_trajectory()
        traj["messages"][7]["content"] = 'selected\n\n可点击的按钮: ["Description", "Buy Now"]'
        navigate = assistant_tool("view_description", {}, "call_navigation")
        traj["messages"][8:8] = [
            navigate,
            tool_message("call_navigation", "view_description", 'details\n\n可点击的按钮: ["Buy Now"]'),
        ]
        traj["steps"].insert(
            3,
            {
                "tool_name": "view_description",
                "tool_call": navigate["tool_calls"][0],
                "env_action": "click[Description]",
                "done": False,
            },
        )

        accepted, reasons = acceptance_reasons(traj)

        self.assertFalse(accepted)
        self.assertIn("step_3.action_not_allowed_after_option_selection", reasons)

    def test_acceptance_reasons_ignores_runtime_guard_when_finding_previous_observation(self):
        traj = accepted_trajectory()
        invalid = assistant_tool("view_features", {}, "blocked_call")
        traj["messages"][6:6] = [
            invalid,
            {
                "role": "tool",
                "tool_call_id": "blocked_call",
                "name": "view_features",
                "content": "未执行：当前页面没有 Features。",
                "runtime_action_guard": True,
            },
        ]
        traj["blocked_tool_calls"] = [{"tool_call": invalid["tool_calls"][0]}]

        accepted, reasons = acceptance_reasons(traj)

        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_build_sft_row_keeps_only_training_messages_and_tools(self):
        traj = accepted_trajectory()
        traj["messages"][2]["reasoning_content"] = "这是 Teacher 的内部推理，只用于 rollout 连贯性。"
        invalid = assistant_tool("view_features", {}, "blocked_call")
        traj["messages"][4:4] = [
            invalid,
            {
                "role": "tool",
                "tool_call_id": "blocked_call",
                "name": "view_features",
                "content": "未执行：当前页面没有 Features。",
                "runtime_action_guard": True,
            },
        ]
        # Guard 后的下一轮推理会复述本地规则，不能进入训练数据；
        # 但它携带的合法工具调用仍必须保留以维持 message/tool 配对。
        traj["messages"][6]["reasoning_content"] = "guard_only_reasoning"
        traj["blocked_tool_calls"] = [{"tool_call": invalid["tool_calls"][0]}]
        traj["messages"][-1]["content"] = "Purchased. Target. Goal. Reward: 1.0. Reward Details."

        row = build_sft_row(traj)

        payload = json.dumps(row, ensure_ascii=False)
        self.assertIn("messages", row)
        self.assertIn("tools", row)
        self.assertNotIn("reward_detail", payload)
        self.assertNotIn('"goal"', payload)
        self.assertNotIn('"purchase"', payload)
        self.assertIn("<think>这是 Teacher 的内部推理，只用于 rollout 连贯性。</think>", payload)
        self.assertNotIn("guard_only_reasoning", payload)
        self.assertNotIn("reasoning_content", payload)
        self.assertNotIn("未执行", payload)
        self.assertEqual(row["messages"][-1]["content"], "购买已完成。")
        self.assertNotIn("Reward", payload)
        self.assertNotIn("Goal", payload)
        self.assertEqual(row["messages"][-2]["tool_calls"][0]["function"]["name"], "buy_now")

    def test_process_raw_trajectories_writes_accepted_rejected_and_sft_jsonl(self):
        bad = accepted_trajectory()
        bad["trajectory_id"] = "traj-bad"
        bad["status"] = "error"
        bad["error"] = {"type": "RuntimeError", "message": "boom"}
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw.jsonl"
            accepted = Path(tmpdir) / "accepted.jsonl"
            rejected = Path(tmpdir) / "rejected.jsonl"
            stats = Path(tmpdir) / "stats.json"
            sft = Path(tmpdir) / "sft.jsonl"
            raw.write_text(
                json.dumps(accepted_trajectory(), ensure_ascii=False)
                + "\n"
                + json.dumps(bad, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            summary = process_raw_trajectories(raw, accepted, rejected, stats, sft)

            accepted_rows = accepted.read_text(encoding="utf-8").splitlines()
            rejected_rows = [json.loads(line) for line in rejected.read_text(encoding="utf-8").splitlines()]
            sft_rows = [json.loads(line) for line in sft.read_text(encoding="utf-8").splitlines()]
            stats_data = json.loads(stats.read_text(encoding="utf-8"))

        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(len(accepted_rows), 1)
        self.assertIn("has_error", rejected_rows[0]["reject_reasons"])
        self.assertEqual(len(sft_rows), 1)
        self.assertEqual(stats_data["total"], 2)


if __name__ == "__main__":
    unittest.main()
