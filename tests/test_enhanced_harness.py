"""CPU integration checks for optional entrypoints and saved-result auditing."""

import asyncio
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts import evaluate_shop_benchmark as cli
from shopping_grpo.evaluation.paired_audit import compare, strict_success, terminal_for_policy
from shopping_grpo.training.grpo.adapter.enhanced_protocol import (
    verify_mask,
)


def row(task, success=True):
    return {
        "task_id": task,
        "done": True,
        "terminal_result": {
            "done": True,
            "over": True,
            "purchase": {"asin": "123456789012"},
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase" if success else "wrong_purchase",
                "reward_valid": True,
            },
        },
    }


class PairedAuditTest(unittest.TestCase):
    def test_complete_purchase_required(self):
        value = row(1)
        self.assertTrue(strict_success(value))
        for key in ("purchase", "done", "over"):
            broken = deepcopy(value)
            broken["terminal_result"].pop(key)
            self.assertFalse(strict_success(broken))

    def test_pairing_rejects_missing_duplicate_and_counts_losses(self):
        result = compare([row(1), row(2, False)], [row(1, False), row(2)])
        self.assertEqual(result["gained"], [2])
        self.assertEqual(result["lost"], [1])
        for other in ([row(1)], [row(1), row(1)]):
            with self.assertRaises(ValueError):
                compare([row(1), row(2)], other)

    def test_original_policy_does_not_mix_reviewed_score(self):
        value = row(1)
        value["terminal_result"]["reward_detail"]["evidence"] = {
            "review_policy_version": "review6",
            "legacy_result": {
                "reward": -0.85,
                "reward_type": "wrong_purchase",
                "reward_valid": True,
            },
        }
        before = deepcopy(value)
        self.assertFalse(strict_success(value, "original"))
        self.assertTrue(strict_success(value, "environment"))
        self.assertEqual(value, before)
        del value["terminal_result"]["reward_detail"]["evidence"]["legacy_result"]
        with self.assertRaises(ValueError):
            terminal_for_policy(value["terminal_result"], "original")

    def test_original_gold_populates_existing_summary_contract(self):
        from shopping_grpo.evaluation.summary import summarize_trajectories

        value = row(1, False)
        value["status"] = "done"
        value["terminal_result"]["reward_detail"]["evidence"] = {
            "review_policy_version": "review6",
            "legacy_result": {"reward": 1.0, "reward_type": "gold_purchase", "reward_valid": True},
        }
        value["terminal_result"] = terminal_for_policy(value["terminal_result"], "original")
        self.assertEqual(summarize_trajectories([1], [value])["strict_successes"], 1)

    def test_loss_mask(self):
        self.assertEqual(verify_mask([1, 2, 3], [1, 0, 1])["generated_tokens"], 2)
        with self.assertRaises(ValueError):
            verify_mask([1], [0])


class EnhancedCliTest(unittest.TestCase):
    def args(self, tmp):
        return [
            "eval",
            "--benchmark",
            "unused",
            "--output",
            str(tmp / "raw.jsonl"),
            "--summary",
            str(tmp / "summary.json"),
            "--model",
            "unused",
            "--llm-base-url",
            "http://unused",
            "--api-key",
            "EMPTY",
        ]

    def test_default_and_explicit_profile_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            prompt = tmp / "prompt.txt"
            prompt.write_text("explicit prompt")
            for enhanced in (False, True):
                args = self.args(tmp)
                if enhanced:
                    args += [
                        "--public-support",
                        "--budget-guard",
                        "--option-labels",
                        "--system-prompt",
                        str(prompt),
                        "--reward-policy",
                        "original",
                    ]
                with (
                    patch.object(sys, "argv", args),
                    patch.object(cli, "load_tasks", return_value=[]),
                    patch.object(cli, "OpenAIChatClient"),
                    patch.object(cli, "collect_tasks") as collect,
                    patch.object(cli, "summarize_trajectories", return_value={}),
                ):
                    cli.main()
                params = collect.call_args.kwargs
                self.assertEqual(params["public_support"], enhanced)
                self.assertEqual(params["budget_guard"], enhanced)
                self.assertEqual(params["system_prompt"], "explicit prompt" if enhanced else None)
                protocol = json.loads((tmp / "summary.json").read_text())["protocol"]
                self.assertEqual(
                    protocol["reward_policy"], "original" if enhanced else "environment"
                )


class EnhancedVerlTest(unittest.TestCase):
    def test_purchase_rejection_does_not_call_environment(self):
        from verl.tools.schemas import OpenAIFunctionToolSchema

        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
        from shopping_grpo.training.grpo.adapter.enhanced_tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.adapter.runtime import (
            current_environment,
            current_runtime_state,
            make_runtime_state,
        )

        async def run():
            state = make_runtime_state(task_id=1, max_steps=35)
            state["task_instruction"] = "不超过10元"
            state["latest_observation"] = (
                "page_type: product_detail\nprice: 20\navailable_options: {}\n"
                'selected_options: {}\n可点击的按钮: ["Buy Now", "Back to Search"]'
            )

            class Env:
                def step(self, action):
                    raise AssertionError("rejected purchase reached environment")

            et = current_environment.set(Env())
            st = current_runtime_state.set(state)
            try:
                schema = next(s for s in SHOP_TOOL_SCHEMAS if s["function"]["name"] == "buy_now")
                tool = ShopSimulatorTool({}, OpenAIFunctionToolSchema.model_validate(schema))
                response, _, _ = await tool.execute("test", {})
                self.assertIn("购买未执行", response.text)
                self.assertEqual(state["steps"], [])
                self.assertEqual(
                    state["frozen_support_events"][0]["reason"], "budget_above_maximum"
                )
            finally:
                current_runtime_state.reset(st)
                current_environment.reset(et)

        asyncio.run(run())

    def test_prompt_replacement_precedes_parent_tokenization(self):
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.enhanced_agent import EnhancedShoppingAgentLoop
        from shopping_grpo.training.grpo.adapter.runtime import current_runtime_state

        loop = object.__new__(EnhancedShoppingAgentLoop)
        loop.system_prompt = "selected enhanced prompt"
        data = SimpleNamespace(
            messages=[
                {"role": "system", "content": "old"},
                {"role": "user", "content": "buy a mug"},
            ],
            extra_fields={},
        )
        state = {"latest_observation": "public search page"}
        token = current_runtime_state.set(state)
        try:
            with patch.object(
                ShoppingToolAgentLoop, "_handle_pending_state", new=AsyncMock(return_value=False)
            ):
                asyncio.run(loop._handle_pending_state(data, {}))
        finally:
            current_runtime_state.reset(token)
        self.assertEqual(data.messages[0]["content"], "selected enhanced prompt")
        self.assertEqual(state["task_instruction"], "buy a mug")
        self.assertEqual(data.messages[-1]["content"], "buy a mug\n\npublic search page")

    def test_nonpurchase_rewards_are_preserved_without_disk_recording(self):
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.enhanced_agent import EnhancedShoppingAgentLoop

        loop = object.__new__(EnhancedShoppingAgentLoop)
        loop.select_original_reward = False
        loop.trajectory_directory = None
        output = SimpleNamespace(
            extra_fields={
                "shopping": {
                    "task_id": 1,
                    "done": False,
                    "termination_reason": "max_steps",
                    "reward": {"total": 0},
                }
            },
            response_ids=[1],
            response_mask=[1],
            reward_score=0,
        )
        for reason in (
            "max_steps",
            "too_many_guard_rejections",
            "parallel_tool_calls",
            "assistant_finished_without_environment_done",
        ):
            for upstream_reward in (0.0, -0.2):
                with self.subTest(reason=reason, reward=upstream_reward):
                    info = output.extra_fields["shopping"]
                    info["termination_reason"] = reason
                    info["reward"] = {"total": upstream_reward}
                    output.reward_score = upstream_reward
                    before = deepcopy(info)
                    with patch.object(
                        ShoppingToolAgentLoop, "run", new=AsyncMock(return_value=output)
                    ):
                        result = asyncio.run(loop.run({}))
                    self.assertEqual(result.reward_score, upstream_reward)
                    self.assertEqual(info["reward"], before["reward"])
                    self.assertEqual(info["termination_reason"], reason)
                    self.assertNotIn("failure_reward_policy", info)
