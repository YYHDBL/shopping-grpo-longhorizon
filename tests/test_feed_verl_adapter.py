"""Dependency-free tests for the Feed veRL 0.8 adapter and preflight."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.schema import EpisodeSeed, Persona, Product, Video
from shopping_grpo.feed.simulator import FeedActionGuardError
from shopping_grpo.feed.tools import FEED_TOOL_SCHEMAS_BY_NAME
from shopping_grpo.feed.verl_adapter import (
    FEED_TOOL_PAYLOAD_VERSION,
    SHOPPING_PATCH_COMPATIBILITY,
    FeedShoppingTool,
    FeedTask,
    FeedTaskStore,
    FeedToolAgentLoop,
    FeedTrajectorySession,
    ToolAgentLoop,
    VERL_AVAILABLE,
    build_process_credit_metadata,
    current_feed_environment,
    current_feed_runtime_state,
    make_feed_runtime_state,
)
from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_tool(name: str) -> FeedShoppingTool:
    schema = FEED_TOOL_SCHEMAS_BY_NAME[name]
    if VERL_AVAILABLE:  # pragma: no cover - exercised by pinned-runtime preflight
        from verl.tools.schemas import OpenAIFunctionToolSchema

        schema = OpenAIFunctionToolSchema.model_validate(schema)
    return FeedShoppingTool({}, schema)


def _fixture_task(video_count: int = 1) -> FeedTask:
    products = [
        Product(
            product_id="p1",
            title="Blue travel mug",
            category="mug",
            price=20.0,
            attributes=("blue", "insulated"),
            rating=4.8,
            review_summary="Keeps drinks warm.",
            complement_product_ids=("p2",),
        ),
        Product(
            product_id="p2",
            title="Mug cleaning brush",
            category="brush",
            price=8.0,
            attributes=("mug", "cleaning"),
            rating=4.5,
        ),
    ]
    videos = tuple(
        Video(
            video_id=f"v{index}",
            caption="blue insulated mug",
            scene=("kitchen",),
            objects=("mug",),
            topics=("mug",),
            related_product_ids=("p1",),
            duration_seconds=20.0,
        )
        for index in range(video_count)
    )
    seed = EpisodeSeed(
        episode_id="episode-1",
        persona=Persona(
            persona_id="persona-1",
            budget=100.0,
            category_interests=("mug",),
            style_preferences=("blue",),
        ),
        videos=videos,
        product_ids=("p1", "p2"),
        seed=7,
        inventory=(("p1", 2), ("p2", 2)),
    )
    return FeedTask(seed, ProductCatalog(products))


class _FakePublicEnv:
    def __init__(self, *, leak: bool = False):
        self.step_index = 0
        self.leak = leak
        self.calls = []
        self._observation = {
            "observation_version": "feed-observation-v1",
            "environment_version": "feed-environment-v1",
            "episode_id": "fake",
            "step": 0,
            "total_steps": 1,
            "persona": {"persona_id": "u", "budget": 100.0},
            "current_video": {"video_id": "v", "caption": "mug"},
            "recent_events": [],
            "cart": [],
            "purchased_product_ids": [],
            "visible_product_ids": ["p1", "p2"],
            "evidence_ids": ["product.p1.price", "video.v.caption"],
            "info_tool_calls": 0,
            "max_info_tool_calls": 3,
            "done": False,
        }

    def observation(self):
        return dict(self._observation)

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.leak:
            return {"reward": 99.0, "result": "must never be rendered"}
        if name == "commit_recommendation":
            self._observation["done"] = True
            self._observation["current_video"] = None
            return {
                "observation": self.observation(),
                "events": [
                    {
                        "event_id": "e1",
                        "event_type": "watch",
                        "step": 0,
                        "source_step": 0,
                        "product_id": None,
                        "value": 12.0,
                        "metadata": {"video_id": "v"},
                    }
                ],
                "done": True,
            }
        return {"items": ["public"], "evidence_ids": ["product.p1.price"]}


class _FakeGuardRejectingEnv(_FakePublicEnv):
    def call_tool(self, name, arguments):
        del name, arguments
        raise FeedActionGuardError("simulator-only guard detail")


class FeedToolFallbackTest(unittest.TestCase):
    def _execute(self, name, arguments, *, env=None):
        async def run():
            bound_env = env or _FakePublicEnv()
            state = make_feed_runtime_state("fake")
            state["latest_observation"] = bound_env.observation()
            env_token = current_feed_environment.set(bound_env)
            state_token = current_feed_runtime_state.set(state)
            try:
                response, reward, metrics = await _make_tool(name).execute(
                    "tool-1", arguments
                )
            finally:
                current_feed_runtime_state.reset(state_token)
                current_feed_environment.reset(env_token)
            return bound_env, state, response, reward, metrics

        return asyncio.run(run())

    def test_all_seven_information_tools_dispatch_through_context_local_env(self):
        calls = {
            "retrieve_products": {"query": "mug"},
            "inspect_product": {"product_id": "p1"},
            "compare_products": {"product_ids": ["p1", "p2"]},
            "read_reviews": {"product_id": "p1"},
            "find_alternatives": {"product_id": "p1"},
            "find_complements": {"product_id": "p1"},
            "check_inventory": {"product_ids": ["p1", "p2"]},
        }
        for name, arguments in calls.items():
            with self.subTest(tool=name):
                env, state, response, reward, metrics = self._execute(name, arguments)
                payload = json.loads(response.text)
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["tool"], name)
                self.assertEqual(payload["payload_version"], FEED_TOOL_PAYLOAD_VERSION)
                self.assertIn("state_delta", payload)
                self.assertIn("result", payload)
                self.assertNotIn("observation", payload)
                self.assertNotIn("persona", payload["state_delta"])
                self.assertNotIn("recent_events", payload["state_delta"])
                self.assertEqual(reward, 0.0)
                self.assertTrue(metrics["accepted"])
                self.assertEqual(env.calls, [(name, arguments)])
                self.assertNotIn("reward", response.text.casefold())
                self.assertNotIn("latent", response.text.casefold())
                self.assertEqual(len(state["tool_calls"]), 1)

    def test_commit_is_non_rewarding_to_model_and_marks_terminal(self):
        arguments = {
            "decision": "recommend",
            "surface": "product_card",
            "product_ids": ["p1"],
            "relationship": "primary",
            "strategy": "direct",
            "evidence_ids": ["product.p1.price", "video.v.caption"],
            "explanation": "grounded recommendation",
        }
        _, state, response, reward, metrics = self._execute(
            "commit_recommendation", arguments
        )
        payload = json.loads(response.text)
        self.assertTrue(payload["result"]["done"])
        self.assertTrue(payload["state_delta"]["done"])
        self.assertIsNone(payload["state_delta"]["current_video"])
        self.assertNotIn("observation", payload["result"])
        self.assertEqual(reward, 0.0)
        self.assertTrue(metrics["done"])
        self.assertTrue(state["done"])
        self.assertTrue(state["terminate"])
        self.assertNotIn("reward", response.text.casefold())

    def test_fail_closed_payload_filter_blocks_reward_and_latent_leaks(self):
        env, state, response, reward, metrics = self._execute(
            "retrieve_products", {"query": "mug"}, env=_FakePublicEnv(leak=True)
        )
        del env
        payload = json.loads(response.text)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "tool_execution_failed")
        self.assertNotIn("reward", response.text.casefold())
        self.assertNotIn("latent", response.text.casefold())
        self.assertEqual(reward, 0.0)
        self.assertFalse(metrics["accepted"])
        self.assertTrue(state["infrastructure_invalid"])

    def test_simulator_guard_mismatch_is_retryable_not_infrastructure_failure(self):
        _, state, response, reward, metrics = self._execute(
            "retrieve_products", {"query": "mug"}, env=_FakeGuardRejectingEnv()
        )
        payload = json.loads(response.text)
        self.assertEqual(payload["error"]["code"], "runtime_guard_rejection")
        self.assertNotIn("simulator-only", response.text)
        self.assertEqual(reward, 0.0)
        self.assertFalse(metrics["accepted"])
        self.assertFalse(state["infrastructure_invalid"])
        self.assertFalse(state["terminate"])
        self.assertEqual(state["guard_rejections"], 1)

    def test_session_binds_and_unbinds_real_feed_environment(self):
        async def run():
            session = FeedTrajectorySession(_fixture_task())
            state = await session.start()
            try:
                self.assertIs(current_feed_environment.get(), session.env)
                self.assertIs(current_feed_runtime_state.get(), state)
                self.assertEqual(
                    state["latest_observation"]["observation_version"],
                    "feed-observation-v1",
                )
            finally:
                await session.close()
            self.assertIsNone(current_feed_environment.get())
            self.assertIsNone(current_feed_runtime_state.get())

        asyncio.run(run())

    def test_48_video_minimal_transcript_uses_bounded_delta_payloads(self):
        async def run():
            session = FeedTrajectorySession(_fixture_task(video_count=48))
            state = await session.start()
            responses = []
            try:
                tool = _make_tool("commit_recommendation")
                for index in range(48):
                    response, reward, metrics = await tool.execute(
                        f"commit-{index}",
                        {
                            "decision": "no_recommend",
                            "surface": "none",
                            "strategy": "none",
                            "relationship": "primary",
                            "product_ids": [],
                            "evidence_ids": [],
                            "explanation": "continue the feed",
                        },
                    )
                    self.assertEqual(reward, 0.0)
                    self.assertTrue(metrics["accepted"])
                    payload = json.loads(response.text)
                    self.assertNotIn("observation", payload["result"])
                    self.assertNotIn("persona", payload["state_delta"])
                    self.assertNotIn("recent_events", payload["state_delta"])
                    responses.append(response.text)
                self.assertTrue(state["done"])
                return responses
            finally:
                await session.close()

        responses = asyncio.run(run())
        # Regression guard for the former quadratic full-observation replay.
        # The 48-video response stream stays linear and well below a conservative
        # character proxy for the 61,440-token generation budget.
        self.assertLess(sum(map(len, responses)), 100_000)
        self.assertLess(max(map(len, responses)), 8_000)

    def test_task_store_resolves_by_registered_key_and_episode_id(self):
        task = _fixture_task()
        store = FeedTaskStore({7: task})
        self.assertIs(store.resolve(7), task)
        self.assertIs(store.resolve("episode-1"), task)
        self.assertEqual(len(store), 1)


class FeedProcessCreditTest(unittest.TestCase):
    def setUp(self):
        self.transitions = [
            {
                "reward": {
                    "qualified_purchase_value": 0.0,
                    "return_penalty": 0.0,
                    "total": 1.0,
                },
                "events": [{"event_type": "watch", "source_step": 0}],
            },
            {
                "reward": {
                    "qualified_purchase_value": 2.0,
                    "return_penalty": 0.0,
                    "satisfaction": 1.0,
                    "total": 3.0,
                },
                "events": [
                    {
                        "event_type": "purchase",
                        "source_step": 0,
                        "metadata": {
                            "qualified_purchase_credit": 2.0,
                            "satisfaction_credit": 1.0,
                        },
                    }
                ],
            },
        ]

    def _credit(self, mode, **kwargs):
        return build_process_credit_metadata(
            self.transitions,
            [2, 7],
            response_length=10,
            mode=mode,
            gamma=0.5,
            episode_return=4.0,
            **kwargs,
        )

    def test_terminal_rtg_event_and_counterfactual_credit_vectors(self):
        self.assertEqual(self._credit("terminal")["step_credit"], [0.0, 4.0])
        self.assertEqual(self._credit("rtg")["step_credit"], [2.5, 3.0])
        self.assertEqual(self._credit("event")["step_credit"], [4.0, 0.0])
        self.assertEqual(
            self._credit(
                "counterfactual", counterfactual_values=[0.25, -0.5]
            )["step_credit"],
            [0.25, -0.5],
        )

    def test_credit_maps_only_to_assistant_commit_positions_and_is_metadata(self):
        credit = self._credit("event")
        expected = [0.0] * 10
        expected[2] = 4.0
        self.assertEqual(credit["response_credit_vector"], expected)
        self.assertTrue(credit["metadata_only"])
        self.assertFalse(credit["native_advantage_integration"])
        self.assertEqual(
            credit["native_reward_boundary"], "scalar_terminal_reward_only"
        )


class FeedAgentLoopSettlementTest(unittest.TestCase):
    def test_terminal_episode_sets_scalar_reward_and_safe_extra_fields(self):
        task = _fixture_task()

        async def fake_parent_run(_loop, sampling_params, **kwargs):
            del sampling_params, kwargs
            env = current_feed_environment.get()
            state = current_feed_runtime_state.get()
            result = env.step(
                {
                    "decision": "no_recommend",
                    "surface": "none",
                    "strategy": "none",
                    "relationship": "primary",
                    "product_ids": [],
                    "evidence_ids": [],
                    "explanation": "wait",
                }
            )
            state["latest_observation"] = result.observation
            state["done"] = True
            state["terminate"] = True
            state["termination_reason"] = "environment_done"
            state["tool_calls"].append({"tool": "commit_recommendation"})
            state["commits"].append({"feed_step": 0, "done": True})
            state["assistant_commit_positions"].append(1)
            return SimpleNamespace(
                prompt_ids=[1],
                response_ids=[2, 3],
                response_mask=[1, 1],
                reward_score=None,
                metrics=object(),
                extra_fields={"preserved": True},
            )

        async def run():
            loop = object.__new__(FeedToolAgentLoop)
            loop.credit_mode = "counterfactual"
            loop.credit_gamma = 0.99
            loop.min_feed_steps = 1
            loop.max_feed_steps = 48
            loop.max_info_calls_per_video = 3
            loop.required_environment_version = "feed-environment-v1"
            loop.required_observation_version = "feed-observation-v1"
            loop.required_reward_version = "feed-reward-v1"
            loop.required_tools_version = "feed-tools-v1"
            loop.task_store = None
            loop.catalog = task.catalog
            loop.catalog_path = None
            loop.calibration = None
            from shopping_grpo.feed.simulator import FeedShoppingEnv

            loop.env_factory = FeedShoppingEnv
            with patch.object(ToolAgentLoop, "run", fake_parent_run):
                output = await FeedToolAgentLoop.run(
                    loop,
                    {},
                    episode_seed=task.episode_seed,
                    raw_prompt=[],
                )
            return output

        output = asyncio.run(run())
        feed = output.extra_fields["feed"]
        self.assertEqual(output.reward_score, feed["episode_return"])
        self.assertTrue(feed["done"])
        self.assertEqual(feed["versions"]["environment"], "feed-environment-v1")
        self.assertEqual(
            feed["credit"]["counterfactual_status"],
            "computed_with_common_random_numbers",
        )
        self.assertEqual(feed["credit"]["step_credit"], [0.0])
        self.assertTrue(output.extra_fields["preserved"])
        shopping = output.extra_fields["shopping"]
        self.assertEqual(
            shopping["compatibility_contract"], SHOPPING_PATCH_COMPATIBILITY
        )
        self.assertEqual(shopping["profile"], "feed")
        self.assertIn("reward/shaped_mean", aggregate_shopping_metrics([shopping]))
        utilities, successes, invalid, reasons = extract_shopping_group_signals(
            [shopping]
        )
        self.assertEqual(utilities, [output.reward_score])
        self.assertEqual(successes, [False])
        self.assertEqual(invalid, [False])
        self.assertEqual(reasons, [()])
        self.assertNotIn("fatigue", json.dumps(feed).casefold())
        self.assertNotIn("trust", json.dumps(feed).casefold())
        self.assertIsNone(current_feed_environment.get())
        self.assertIsNone(current_feed_runtime_state.get())


class FeedConfigPreflightTest(unittest.TestCase):
    def test_static_preflight_checks_configs_without_starting_training(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_feed_grpo_runtime.py"),
                "--root",
                str(ROOT),
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        report = json.loads(completed.stdout)
        self.assertTrue(report["ok"])
        self.assertFalse(report["training_started"])
        self.assertEqual(report["config"]["tool_count"], 8)
        self.assertGreaterEqual(report["config"]["max_assistant_turns"], 128)
        self.assertTrue(
            report["capability_boundary"]["requires_trainer_patch_for_process_advantage"]
        )


if __name__ == "__main__":
    unittest.main()
