"""不依赖 veRL 安装的最小适配层单测。"""

import asyncio
import threading
import unittest
from unittest.mock import patch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool


def make_tool(name):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test-only {name} tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        tool_schema = schema
    else:
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, tool_schema)


class VerlAdapterRuntimeTest(unittest.TestCase):
    def test_agent_loop_preserves_real_verl_metrics_and_exports_shopping_diagnostics(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "shopsimulator-environment-v2.1",
                }

            def release(self):
                self.released = True

        async def fake_parent_run(_loop, sampling_params, **kwargs):
            state = current_runtime_state.get()
            state.update(
                {
                    "done": True,
                    "terminal_result": {"done": True, "over": True},
                    "termination_reason": "gold_purchase",
                    "final_reward": 1.0,
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {"key_options": 1.0},
                        "hard_gates": {
                            "category": {"passed": True},
                            "budget": {"passed": True},
                        },
                    },
                    "steps": [
                        {
                            "index": 0,
                            "tool": "search",
                            "parameters": {"query": "shoe"},
                            "done": False,
                            "reward": 0.0,
                        }
                    ],
                    "guard_rejection_reason_counts": {"click_not_in_previous_observation": 2},
                }
            )
            return AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
                reward_score=None,
                metrics=AgentLoopMetrics(generate_sequences=0.25),
                extra_fields={},
            )

        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.base_url = "http://shop.test"
            loop.timeout = 60
            loop.max_steps = 35
            loop.required_environment_version = "shopsimulator-environment-v2.1"
            loop.reward_mode = "constraint_aware"
            loop.env_factory = FakeEnv
            with patch.object(ToolAgentLoop, "run", fake_parent_run):
                return await ShoppingToolAgentLoop.run(
                    loop,
                    {},
                    extra_info={"task_id": 42},
                )

        output = asyncio.run(run())
        self.assertIsInstance(output.metrics, AgentLoopMetrics)
        self.assertEqual(
            output.metrics.model_dump(),
            {
                "generate_sequences": 0.25,
                "tool_calls": 0.0,
                "compute_score": 0.0,
                "num_preempted": -1,
            },
        )
        self.assertEqual(output.reward_score, 1.0)
        self.assertEqual(output.extra_fields["shopping"]["task_id"], 42)
        self.assertEqual(
            output.extra_fields["shopping"]["reward"]["terminal_utility"],
            1.0,
        )
        self.assertEqual(
            output.extra_fields["shopping"]["actions"],
            [{"tool": "search", "parameters": {"query": "shoe"}}],
        )
        self.assertEqual(
            output.extra_fields["shopping"]["guard_rejection_reasons"],
            {"click_not_in_previous_observation": 2},
        )
        self.assertTrue(created[0].released)

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
            self.assertTrue(state["infrastructure_invalid"])
            self.assertIsNone(state["reward_detail"])
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_terminal_reward_components_are_validated_without_entering_tool_observation(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "Goal: hidden answer",
                    "done": True,
                    "over": True,
                    "reward": 0.6,
                    "goal": {"secret": True},
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "partial_alternative_purchase",
                        "termination_reason": "partial_alternative_purchase",
                        "reward_valid": True,
                        "terminal_utility": 0.6,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "hard_gates": {},
                        "hidden_answer": "do not retain",
                    },
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
            self.assertFalse(state["infrastructure_invalid"])
            self.assertEqual(
                state["reward_type"], "partial_alternative_purchase")
            self.assertTrue(state["reward_valid"])
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_terminal_reward_keeps_unverifiable_separate_from_infrastructure(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.0,
                    "termination_reason": "reward_unverifiable",
                    "reward_valid": False,
                    "reward_detail": {
                        "reward_version": "unsupported-reward",
                        "reward_type": "reward_unverifiable",
                        "reward_valid": False,
                        "termination_reason": "reward_unverifiable",
                        "target_asin_match": False,
                        "hard_gates": {
                            "category": {"passed": True, "verifiable": True}
                        },
                        "weighted_score": 0.0,
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute(
                    "tool-v2", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertTrue(state["reward_unverifiable"])
            self.assertEqual(state["reward_type"], "reward_unverifiable")
            self.assertEqual(state["termination_reason"], "reward_unverifiable")

        asyncio.run(run())

    def test_reward_exposes_utility_success_and_sampling_validity_separately(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.55,
                    "termination_reason": "valid_alternative_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "valid_alternative_purchase",
                        "reward_valid": True,
                        "termination_reason": "valid_alternative_purchase",
                        "target_asin_match": False,
                        "terminal_utility": 0.55,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {
                            "brand": 0.0,
                            "model": 0.0,
                            "core_functions": 1.0,
                            "key_options": 1.0,
                        },
                        "hard_gates": {
                            "category": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                                "comparator": "category_leaf_ancestor_chain",
                                "source_field": "category",
                            }
                        },
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute(
                    "tool-v3",
                    {"query": "mug"},
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertFalse(state["reward_unverifiable"])
            self.assertEqual(
                state["reward_type"],
                "valid_alternative_purchase",
            )
            breakdown = reward_breakdown(state)
            self.assertEqual(breakdown["terminal_utility"], 0.55)
            self.assertEqual(breakdown["purchase_success"], 1.0)
            self.assertEqual(breakdown["r_att"], 1.0)
            self.assertEqual(breakdown["r_option"], 1.0)
            self.assertFalse(breakdown["sampling_invalid"])

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

    def test_repeated_guard_rejections_terminate_instead_of_looping_forever(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "可点击的按钮: []"
            state["latest_observation_truncated"] = True
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
            self.assertEqual(state["steps"], [])
            self.assertEqual(state["action_attempt_count"], 3)
            self.assertEqual(state["repeat_action_count"], 2)
            self.assertEqual(state["guard_rejection_count"], 3)
            self.assertEqual(state["guard_rejection_after_truncation_count"], 3)
            self.assertEqual(state["action_attempt_after_truncation_count"], 3)
            self.assertEqual(
                state["guard_rejection_reason_counts"],
                {"click_not_in_previous_observation": 3},
            )
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

    def test_session_rejects_wrong_environment_version_and_releases(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "unsupported-environment",
                }

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(
                required_environment_version="shopsimulator-environment-v2.1",
                env_factory=FakeEnv,
            )
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                await session.start(1)

        asyncio.run(run())
        self.assertTrue(created[0].released)

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


class RewardV31Test(unittest.TestCase):
    """v3.1：截断档 -0.5；连续长度惩罚不设（用户裁决 2026-09-29）。"""

    def _truncated_state(self) -> dict:
        state = make_runtime_state(task_id=9, max_steps=35)
        state["done"] = False
        state["error"] = "assistant_finished_without_environment_done"
        state["steps"] = [
            {"tool": "search_products", "parameters": {"query": "mug"}}
        ] * 35
        return state

    def test_truncation_gets_discrete_penalty_in_both_modes(self):
        state = self._truncated_state()
        breakdown = reward_breakdown(state)
        self.assertEqual(breakdown["total"], -0.5)
        self.assertTrue(breakdown["truncated"])
        self.assertFalse(breakdown["sampling_invalid"])
        self.assertEqual(terminal_reward(state, mode="native"), -0.5)
        self.assertEqual(terminal_reward(state, mode="constraint_aware"), -0.5)

    def test_truncation_with_infra_invalid_stays_zero(self):
        state = self._truncated_state()
        state["infrastructure_invalid"] = True
        self.assertEqual(reward_breakdown(state)["total"], 0.0)
        self.assertEqual(terminal_reward(state, mode="native"), 0.0)

    def test_normal_terminal_state_unaffected_by_truncation_branch(self):
        # 正常终局（非截断）不该被误伤：total 仍取环境 native 值
        state = make_runtime_state(task_id=9, max_steps=35)
        state["done"] = True
        state["terminal_result"] = {"done": True, "over": True}
        state["final_reward"] = 1.0
        state["reward_valid"] = True
        state["reward_version"] = "shopsimulator-reward-v3"
        state["reward_type"] = "gold_purchase"
        state["reward_detail"] = {"hard_gates": {}, "dimension_scores": {}}
        self.assertFalse(reward_breakdown(state)["truncated"])
        self.assertEqual(reward_breakdown(state)["total"], 1.0)

    def test_long_but_normal_terminal_has_no_length_penalty(self):
        # 连续长度惩罚不设：35 步正常终局（gold）拿满分，不被步数扣分
        state = make_runtime_state(task_id=9, max_steps=35)
        state["done"] = True
        state["terminal_result"] = {"done": True, "over": True}
        state["final_reward"] = 1.0
        state["reward_valid"] = True
        state["reward_version"] = "shopsimulator-reward-v3"
        state["reward_type"] = "gold_purchase"
        state["reward_detail"] = {"hard_gates": {}, "dimension_scores": {}}
        state["steps"] = [{"tool": "search_products", "parameters": {}}] * 35
        self.assertEqual(reward_breakdown(state)["total"], 1.0)


class GroupStatsTest(unittest.TestCase):
    """组级 reward 统计纯函数（compat.compute_group_stats）。"""

    def test_zero_variance_and_std(self):
        from shopping_grpo.training.grpo.compat import compute_group_stats

        # 组 A：8 条全 1.0（零方差）；组 B：4 条 0.25 / 4 条 -0.85（有大方差）
        uids = [f"task-a_0_{i}" for i in range(8)] + [f"task-b_0_{i}" for i in range(8)]
        scores = [1.0] * 8 + [0.25] * 4 + [-0.85] * 4
        stats = compute_group_stats(scores, uids)
        self.assertEqual(stats["group/zero_variance_ratio"], 0.5)
        # 组 B 的 std=0.55，组 A 的 std=0，均值 0.275
        self.assertAlmostEqual(stats["group/reward_std_mean"], 0.275, places=3)

    def test_single_member_groups_skipped(self):
        from shopping_grpo.training.grpo.compat import compute_group_stats

        stats = compute_group_stats([1.0, 0.0], ["t1_0_0", "t2_0_0"])
        self.assertEqual(stats, {})

    def test_rollout_suffix_separates_groups(self):
        from shopping_grpo.training.grpo.compat import compute_group_stats

        # 真实结构（run2 smoke 落盘实测）：中间段是 sample 序号 0..7，
        # 组 ID = task uuid（去末两段）
        uids = [f"t1_{i}_0" for i in range(8)] + [f"t2_{i}_0" for i in range(8)]
        stats = compute_group_stats([1.0] * 8 + [0.25] * 4 + [-0.85] * 4, uids)
        self.assertEqual(stats["group/zero_variance_ratio"], 0.5)


class LazyPatchTest(unittest.TestCase):
    """patch_module_after_import 与延迟补丁的应用逻辑（不依赖 veRL）。"""

    def _write_module(self, tmpdir, name, body):
        import pathlib

        path = pathlib.Path(tmpdir) / f"{name}.py"
        path.write_text(body, encoding="utf-8")
        return str(path)

    def test_patch_applies_on_import_and_already_imported(self):
        import importlib
        import sys
        import tempfile

        from shopping_grpo.training.grpo.compat import patch_module_after_import

        with tempfile.TemporaryDirectory() as td:
            sys.path.insert(0, td)
            # 三个文件先写齐：importlib 的目录缓存按 mtime 失效，
            # import 之后再写同目录新文件可能因同秒 mtime 读不到
            self._write_module(td, "lp_mod_a", "VALUE = 1\n")
            self._write_module(td, "lp_mod_b", "VALUE = 2\n")
            self._write_module(td, "lp_mod_c", "VALUE = 3\n")
            try:
                # 路径一：注册后首次导入时应用
                patch_module_after_import("lp_mod_a", lambda m: setattr(m, "PATCHED", True))
                mod_a = importlib.import_module("lp_mod_a")
                self.assertTrue(mod_a.PATCHED)
                self.assertEqual(mod_a.VALUE, 1)  # 模块体正常执行
                # 路径二：已导入的模块直接应用
                mod_b = importlib.import_module("lp_mod_b")
                patch_module_after_import("lp_mod_b", lambda m: setattr(m, "PATCHED", True))
                self.assertTrue(mod_b.PATCHED)
                # 未注册的第三方导入不受影响
                mod_c = importlib.import_module("lp_mod_c")
                self.assertFalse(hasattr(mod_c, "PATCHED"))
            finally:
                sys.path.remove(td)
                for name in ("lp_mod_a", "lp_mod_b", "lp_mod_c"):
                    sys.modules.pop(name, None)

    def test_save_and_stop_wraps_trainer(self):
        import os
        import tempfile
        import types

        from shopping_grpo.training.grpo.compat import _apply_save_and_stop

        with tempfile.TemporaryDirectory() as td:
            flag = os.path.join(td, "SAVE_AND_STOP")
            calls = []

            class FakeTrainer:
                def step(self):
                    return "orig_step"

                def _compute_advantage(self, batch, metrics):
                    return "orig_adv"

                def _save_checkpoint(self):
                    calls.append("saved")

            module = types.SimpleNamespace(PPOTrainer=FakeTrainer)
            _apply_save_and_stop(module, flag_path=flag)

            # 无标志：两个方法透传
            t = FakeTrainer()
            self.assertEqual(t.step(), "orig_step")
            self.assertEqual(t._compute_advantage({}, {}), "orig_adv")

            # 有标志：保存、删标志、SystemExit 退出
            open(flag, "w").close()
            with self.assertRaises(SystemExit):
                t.step()
            self.assertEqual(calls, ["saved"])
            self.assertFalse(os.path.exists(flag))
            # 标志已删，后续调用恢复透传
            self.assertEqual(t.step(), "orig_step")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
