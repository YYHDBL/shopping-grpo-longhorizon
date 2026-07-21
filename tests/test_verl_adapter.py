"""不依赖 veRL 安装的最小适配层单测。"""

import asyncio
import unittest

from shopping_grpo.verl_adapter.interaction import ShopSimulatorInteraction
from shopping_grpo.verl_adapter.runtime import (
    current_runtime_state,
    make_runtime_state,
    terminal_reward,
)


class VerlAdapterRuntimeTest(unittest.TestCase):
    def test_terminal_reward_only_uses_a_normal_environment_completion(self):
        done = make_runtime_state(task_id=1, max_steps=35)
        done.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 0.75})
        self.assertEqual(terminal_reward(done), 0.75)

        unfinished = make_runtime_state(task_id=1, max_steps=35)
        unfinished.update({"final_reward": 1.0, "terminal_result": {"done": False}})
        self.assertEqual(terminal_reward(unfinished), 0.0)

    def test_context_state_is_task_local(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        token = current_runtime_state.set(state)
        try:
            self.assertIs(current_runtime_state.get(), state)
        finally:
            current_runtime_state.reset(token)

    def test_runtime_state_has_no_hidden_goal_fields(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        self.assertEqual(set(state), {"task_id", "max_steps", "steps", "done", "terminal_result", "final_reward", "error"})

    def test_interaction_releases_its_environment_on_finalize(self):
        """无论正常终局还是异常路径，veRL lifecycle 都必须归还 ShopSimulator 租约。"""
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {"instruction": f"task {task_id}"}

            def release(self):
                self.released = True

        async def run():
            interaction = ShopSimulatorInteraction({"max_steps": 35}, env_factory=FakeEnv)
            await interaction.start_interaction("trajectory-1", task_id=8)
            state = interaction._instances["trajectory-1"]["state"]
            state.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 1.0})
            self.assertEqual(await interaction.calculate_score("trajectory-1"), 1.0)
            await interaction.finalize_interaction("trajectory-1")

        asyncio.run(run())
        self.assertTrue(created[0].released)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
