import asyncio
import unittest

from shopping_grpo.verl_shop_context import (
    CURRENT_SHOP_ENV,
    CURRENT_SHOP_STATE,
    make_initial_state,
)
from shopping_grpo.verl_shop_tools import Shop_search_products_Tool, Shop_think_Tool


class FakeEnv:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return {
            "observation": "search result",
            "reward": 0.25,
            "done": False,
            "status_code": 200,
        }


class StructuredFakeEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        return {
            "instruction": "structured search result",
            "reward": 0.0,
            "done": False,
        }


class VerlShopToolsTest(unittest.TestCase):
    def test_search_products_executes_env_step(self):
        env = FakeEnv()
        state = make_initial_state(task_id=0)
        env_token = CURRENT_SHOP_ENV.set(env)
        state_token = CURRENT_SHOP_STATE.set(state)
        try:
            tool = Shop_search_products_Tool({}, {"function": {"name": "search_products"}})
            response, reward, info = asyncio.run(
                tool.execute("instance-1", {"query": "乳胶枕"})
            )
        finally:
            CURRENT_SHOP_STATE.reset(state_token)
            CURRENT_SHOP_ENV.reset(env_token)

        self.assertEqual(env.actions, ["search[乳胶枕]"])
        self.assertEqual(response.text, "search result")
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["inc_reward"], 0.25)
        self.assertEqual(state["num_tool_calls"], 1)

    def test_think_records_without_env_step(self):
        env = FakeEnv()
        state = make_initial_state(task_id=0)
        env_token = CURRENT_SHOP_ENV.set(env)
        state_token = CURRENT_SHOP_STATE.set(state)
        try:
            tool = Shop_think_Tool({}, {"function": {"name": "think"}})
            response, reward, info = asyncio.run(
                tool.execute("instance-1", {"note": "need compare options"})
            )
        finally:
            CURRENT_SHOP_STATE.reset(state_token)
            CURRENT_SHOP_ENV.reset(env_token)

        self.assertEqual(env.actions, [])
        self.assertEqual(response.text, "need compare options")
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["tool"], "think")
        self.assertEqual(state["num_tool_calls"], 1)

    def test_search_products_returns_structured_api_instruction(self):
        env = StructuredFakeEnv()
        state = make_initial_state(task_id=0)
        env_token = CURRENT_SHOP_ENV.set(env)
        state_token = CURRENT_SHOP_STATE.set(state)
        try:
            tool = Shop_search_products_Tool({}, {"function": {"name": "search_products"}})
            response, _, _ = asyncio.run(tool.execute("instance-1", {"query": "乳胶枕"}))
        finally:
            CURRENT_SHOP_STATE.reset(state_token)
            CURRENT_SHOP_ENV.reset(env_token)

        self.assertEqual(response.text, "structured search result")


if __name__ == "__main__":
    unittest.main()
