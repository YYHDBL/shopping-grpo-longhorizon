import asyncio
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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
                tool = ShopSimulatorTool(
                    {"purchase_checks": True}, OpenAIFunctionToolSchema.model_validate(schema)
                )
                response, _, _ = await tool.execute("test", {})
                self.assertIn("购买未执行", response.text)
                self.assertEqual(state["steps"], [])
                self.assertEqual(
                    state["frozen_support_events"][-1]["reason"], "budget_above_maximum"
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

    def test_purchase_and_interface_switches_execute_independently(self):
        from verl.tools.schemas import OpenAIFunctionToolSchema

        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
        from shopping_grpo.training.grpo.adapter.enhanced_tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.adapter.runtime import (
            current_environment,
            current_runtime_state,
            make_runtime_state,
        )

        async def run(purchase, interface, observation=None):
            state = make_runtime_state(task_id=1, max_steps=35)
            state.update(
                task_instruction="不超过10元",
                latest_observation=observation
                or (
                    "page_type: product_detail\nprice: 20\navailable_options: {}\n"
                    'selected_options: {}\n可点击的按钮: ["Buy Now"]'
                ),
            )
            calls = []

            class Env:
                def step(self, action):
                    calls.append(action)
                    return {"done": False, "observation": "unchanged"}

            et, st = current_environment.set(Env()), current_runtime_state.set(state)
            try:
                schema = next(s for s in SHOP_TOOL_SCHEMAS if s["function"]["name"] == "buy_now")
                tool = ShopSimulatorTool(
                    {"purchase_checks": purchase, "interface_support": interface},
                    OpenAIFunctionToolSchema.model_validate(schema),
                )
                await tool.execute("test", {})
                self.assertEqual(len(calls), 0 if purchase else 1)
                self.assertEqual(state["guard_rejection_count"], int(purchase))
                if observation is not None:
                    self.assertEqual(
                        state["guard_rejection_reason_counts"], {"variant_state_unreadable": 1}
                    )
                    self.assertEqual(state["steps"], [])
            finally:
                current_runtime_state.reset(st)
                current_environment.reset(et)

        for purchase in (False, True):
            for interface in (False, True):
                asyncio.run(run(purchase, interface))

        for interface in (False, True):
            for options in (
                "selected_options: {}",
                "available_options: {}",
                'available_options: {"颜色":\nselected_options: {}',
                'available_options: {"颜色":["红"]}\nselected_options: invalid',
            ):
                observation = (
                    "page_type: product_detail\nprice: 5\n"
                    + options
                    + '\n可点击的按钮: ["Buy Now"]'
                )
                asyncio.run(run(True, interface, observation))
