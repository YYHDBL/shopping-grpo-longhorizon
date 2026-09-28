# build_sft_dataset 准入与 mask 规则测试（纯 CPU，不需要 tokenizer）：
#   1) 终止审计硬校验：reward_valid/over/终止原因缺失即拒绝；
#   2) 非法工具调用参数（非法 JSON / null / 空串）拒绝且不修复；
#   3) 守卫拒绝的 assistant 段 train=0，正常段 train=1；
#   4) JEV 结论与终止审计字段进入样本，替代购买可独立追溯。
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sft_dataset import (  # noqa: E402
    admission_problem,
    admission_audit,
    build_sample,
    tool_call_argument_problem,
)


def _terminal(reward_type="gold_purchase", reward_valid=True, over=True,
              termination_reason="purchase", purchase=None):
    return {
        "over": over,
        "reward_valid": reward_valid,
        "termination_reason": termination_reason,
        "purchase": purchase if purchase is not None else {
            "name": "商品A", "price": 100.0, "options": {"颜色": "金"},
            "instruction_text": "买个商品A",
        },
        "reward_detail": {
            "reward_type": reward_type,
            "reward_valid": reward_valid,
        },
    }


def _trajectory(messages=None, reward_type="gold_purchase", **kwargs):
    return {
        "messages": messages or [],
        "terminal_result": _terminal(reward_type=reward_type, **kwargs),
        "meta": {
            "record_id": "R1",
            "difficulty": "easy",
            "persona_condition": "no_profile",
            "steps": 3,
            "accepted": True,
            "accept_reason": "gold" if reward_type == "gold_purchase" else "jev:fully_satisfies",
            "reward_type": reward_type,
        },
    }


class ArgumentValidationTest(unittest.TestCase):
    def test_valid_arguments_pass(self):
        call = {"function": {"name": "search_products",
                             "arguments": '{"query": "枕头"}'}}
        self.assertIsNone(tool_call_argument_problem(call))

    def test_null_argument_value_rejected(self):
        call = {"function": {"name": "view_features",
                             "arguments": '{"__dummy__": null}'}}
        self.assertIn("argument_value_null", tool_call_argument_problem(call))

    def test_empty_argument_value_rejected(self):
        call = {"function": {"name": "buy_now", "arguments": '{"_": ""}'}}
        self.assertIn("argument_value_empty", tool_call_argument_problem(call))

    def test_invalid_json_rejected(self):
        call = {"function": {"name": "buy_now", "arguments": "{oops"}}
        self.assertIn("arguments_invalid_json", tool_call_argument_problem(call))

    def test_missing_arguments_rejected(self):
        call = {"function": {"name": "buy_now"}}
        self.assertIn("arguments_missing", tool_call_argument_problem(call))


class AdmissionTest(unittest.TestCase):
    def test_clean_gold_trajectory_passes(self):
        self.assertIsNone(admission_problem(_trajectory()))

    def test_reward_invalid_rejected(self):
        problem = admission_problem(_trajectory(reward_valid=False))
        self.assertEqual(problem, "reward_invalid")

    def test_not_over_rejected(self):
        problem = admission_problem(_trajectory(over=False))
        self.assertEqual(problem, "terminal_not_over")

    def test_missing_terminal_rejected(self):
        trajectory = _trajectory()
        trajectory["terminal_result"] = {}
        self.assertEqual(admission_problem(trajectory), "terminal_incomplete")

    def test_invalid_arguments_reject_otherwise_valid_trajectory(self):
        trajectory = _trajectory(messages=[
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {
                    "name": "view_features",
                    "arguments": '{"__dummy__": null}'}}]},
        ])
        self.assertTrue(
            admission_problem(trajectory).startswith("invalid_arguments:"))

    def test_jev_audit_fields_persisted_for_alternative(self):
        verdict = {"choice": "fully_satisfies", "probabilities": {"fully_satisfies": 0.9},
                   "confidence": 0.9, "model_reported": "m", "provider": "p",
                   "generation_id": "g1", "request_hash": "h", "rubric_version": "v1"}
        trajectory = _trajectory(reward_type="partial_alternative_purchase")
        trajectory["meta"]["audit"] = {"jev_verdict": verdict}
        sample = build_sample(trajectory, task={}, persona_pool={})
        self.assertEqual(sample["accept_reason"], "jev:fully_satisfies")
        self.assertEqual(sample["reward_type"], "partial_alternative_purchase")
        self.assertTrue(sample["reward_valid"])
        self.assertTrue(sample["over"])
        self.assertEqual(sample["termination_reason"], "purchase")
        self.assertEqual(sample["jev_verdict"], verdict)
        self.assertEqual(sample["purchase"]["name"], "商品A")


class GuardTrainFlagTest(unittest.TestCase):
    def test_guard_rejected_assistant_segment_not_trained(self):
        trajectory = _trajectory(messages=[
            {"role": "user", "content": "买个商品"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {
                    "name": "search_products",
                    "arguments": '{"query": "枕头"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "结果页"},
            {"role": "assistant", "tool_calls": [
                {"id": "c2", "function": {
                    "name": "view_features",
                    "arguments": '{"__dummy__": "x"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "守卫拒绝",
             "runtime_action_guard": True},
            {"role": "assistant", "tool_calls": [
                {"id": "c3", "function": {
                    "name": "buy_now", "arguments": "{}"}}]},
        ])
        sample = build_sample(trajectory, task={}, persona_pool={})
        self.assertEqual(sample["guard_excluded_segments"], 1)
        by_call = {}
        for message in sample["messages"]:
            for call in message.get("tool_calls") or []:
                by_call[call["id"]] = message["train"]
        # 被拒绝的 c2 段不训练；正常 c1/c3 段训练。
        self.assertEqual(by_call, {"c1": 1, "c2": 0, "c3": 1})

    def test_jev_audit_falls_back_to_terminal_without_meta_audit(self):
        sample = build_sample(_trajectory(
            reward_type="valid_alternative_purchase"), task={}, persona_pool={})
        self.assertEqual(sample["accept_reason"], "jev:fully_satisfies")
        self.assertIsNone(sample["jev_verdict"])
        self.assertEqual(sample["purchase"]["price"], 100.0)


if __name__ == "__main__":
    unittest.main()
