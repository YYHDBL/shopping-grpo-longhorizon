"""不依赖 veRL 安装的最小适配层单测。"""

import asyncio
import threading
import unittest

from shopping_grpo.verl_adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.verl_adapter.session import ShopSimulatorSession
from shopping_grpo.verl_adapter.tools import ShopSimulatorTool


def make_tool(name):
    return ShopSimulatorTool({}, {"function": {"name": name}})


class VerlAdapterRuntimeTest(unittest.TestCase):
    def test_terminal_reward_only_uses_a_normal_environment_completion(self):
        done = make_runtime_state(task_id=1, max_steps=35)
        done.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 0.75})
        self.assertEqual(terminal_reward(done), 0.75)

        unfinished = make_runtime_state(task_id=1, max_steps=35)
        unfinished.update({"final_reward": 1.0, "terminal_result": {"done": False}})
        self.assertEqual(terminal_reward(unfinished), 0.0)

        errored = make_runtime_state(task_id=1, max_steps=35)
        errored.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "final_reward": 1.0,
                "error": "tool_error:timeout",
            }
        )
        self.assertEqual(terminal_reward(errored), 0.0)

    def test_context_state_is_task_local(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        token = current_runtime_state.set(state)
        try:
            self.assertIs(current_runtime_state.get(), state)
        finally:
            current_runtime_state.reset(token)

    def test_runtime_state_has_no_hidden_goal_fields(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        self.assertNotIn("goal", state)
        self.assertNotIn("reward_detail", state)

    def test_task_id_is_read_from_verl_extra_info(self):
        self.assertEqual(task_id_from_kwargs({"extra_info": {"task_id": 42}}), 42)

    def test_task_id_accepts_numpy_style_scalar_container(self):
        class Scalar:
            def item(self):
                return {"task_id": 43}

        self.assertEqual(task_id_from_kwargs({"extra_info": Scalar()}), 43)

    def test_missing_task_id_fails_before_acquiring_an_environment(self):
        with self.assertRaisesRegex(ValueError, "task_id"):
            task_id_from_kwargs({"extra_info": {"split": "train"}})

    def test_terminal_observation_is_not_returned_to_the_model(self):
        class FakeEnv:
            def step(self, action):
                self.action = action
                return {
                    "instruction": "Goal: hidden answer\nReward: hidden breakdown",
                    "done": True,
                    "over": True,
                    "reward": 1.0,
                    "goal": {"secret": True},
                    "reward_detail": {"secret": True},
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("search_products").execute(
                    "tool-1", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(response.text, "Environment terminated.")
            self.assertTrue(state["terminate"])
            self.assertEqual(state["terminal_result"], {"done": True, "over": True})
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_sync_environment_step_runs_off_the_event_loop_thread(self):
        main_thread = threading.get_ident()

        class FakeEnv:
            step_thread = None

            def step(self, action):
                self.step_thread = threading.get_ident()
                return {"instruction": "next", "done": False, "over": False, "reward": 0.0}

        async def run():
            env = FakeEnv()
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(env)
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute("tool-1", {"query": "mug"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertNotEqual(env.step_thread, main_thread)

        asyncio.run(run())

    def test_think_consumes_the_step_budget_and_terminates_at_the_exact_limit(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=1)
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("think").execute("tool-1", {"note": "plan"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(len(state["steps"]), 1)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "max_steps")
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_repeated_guard_rejections_terminate_instead_of_looping_forever(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "可点击的按钮: []"
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                tool = make_tool("open_product")
                for index in range(3):
                    response, _, _ = await tool.execute(f"tool-{index}", {"asin": "123456789012"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "too_many_guard_rejections")
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_session_releases_its_environment_on_close(self):
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
            session = ShopSimulatorSession(max_steps=35, env_factory=FakeEnv)
            state = await session.start(task_id=8)
            state.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 1.0})
            self.assertEqual(terminal_reward(state), 1.0)
            await session.close()

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_session_reset_and_release_run_off_the_event_loop_thread(self):
        main_thread = threading.get_ident()
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.reset_thread = None
                self.release_thread = None
                created.append(self)

            def reset(self, task_id):
                self.reset_thread = threading.get_ident()
                return {"instruction": f"task {task_id}"}

            def release(self):
                self.release_thread = threading.get_ident()

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            await session.close()

        asyncio.run(run())
        self.assertNotEqual(created[0].reset_thread, main_thread)
        self.assertNotEqual(created[0].release_thread, main_thread)

    def test_reset_failure_still_releases_the_environment(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                raise RuntimeError("reset failed")

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                await session.start(task_id=8)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_release_failure_is_not_silently_hidden_or_forgotten(self):
        class FakeEnv:
            def __init__(self, **kwargs):
                pass

            def reset(self, task_id):
                return {"instruction": f"task {task_id}"}

            def release(self):
                raise RuntimeError("release failed")

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await session.close()
            self.assertEqual(session.state["error"], "release_error:RuntimeError:release failed")

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
