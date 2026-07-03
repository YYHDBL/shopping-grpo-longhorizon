import asyncio
import unittest

from shopping_grpo.verl_shop_context import CURRENT_SHOP_ENV, CURRENT_SHOP_STATE
from shopping_grpo.verl_shop_interaction import ShopInteraction


class FakeEnv:
    def __init__(self, base_url=None):
        self.base_url = base_url
        self.reset_task_ids = []

    def reset(self, task_id):
        self.reset_task_ids.append(task_id)
        return {"observation": "instruction text", "reward": 0.0, "done": False}


class ShopInteractionTest(unittest.TestCase):
    def test_start_interaction_binds_env_and_state(self):
        interaction = ShopInteraction({"env_factory": FakeEnv, "base_url": "http://fake"})

        async def scenario():
            instance_id = await interaction.start_interaction("i-1", task_id=2)
            return instance_id, CURRENT_SHOP_ENV.get(), CURRENT_SHOP_STATE.get()

        instance_id, env, state = asyncio.run(scenario())

        self.assertEqual(instance_id, "i-1")
        self.assertIsInstance(env, FakeEnv)
        self.assertEqual(state["task_id"], 2)
        self.assertIn("i-1", interaction._instance_dict)

    def test_calculate_score_uses_total_reward(self):
        interaction = ShopInteraction({"env_factory": FakeEnv})
        asyncio.run(interaction.start_interaction("i-1", task_id=0))
        interaction._instance_dict["i-1"]["state"]["total_reward"] = 0.7

        score = asyncio.run(interaction.calculate_score("i-1"))

        self.assertEqual(score, 0.7)

    def test_finalize_removes_instance(self):
        interaction = ShopInteraction({"env_factory": FakeEnv})
        asyncio.run(interaction.start_interaction("i-1", task_id=0))

        asyncio.run(interaction.finalize_interaction("i-1"))

        self.assertNotIn("i-1", interaction._instance_dict)


if __name__ == "__main__":
    unittest.main()
