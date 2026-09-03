"""CPU-only checks for TRACE shopping credit assignment."""

from __future__ import annotations

import math
import asyncio
import unittest

from shopping_grpo.training.grpo.trace import (
    canonical_purchase_target,
    mixed_token_advantages,
    trace_turn_layout,
    turn_rewards,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from scripts.check_grpo_runtime import validate_trace


class TraceCreditTest(unittest.TestCase):
    def test_trace_preflight_requires_the_frozen_lora_base(self):
        config = {
            "shopping_trace": {
                "enable": True,
                "epsilon": 0.1,
                "horizon": 3,
                "discount": 0.8,
                "terminal_weight": 2.0,
                "outcome_weight": 1.0,
                "turn_weight": 0.2,
                "max_sequence_length": 24576,
            },
            "algorithm": {"adv_estimator": "grpo"},
            "actor_rollout_ref": {"model": {"lora_rank": 0}},
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
        }
        with self.assertRaisesRegex(SystemExit, "LoRA"):
            validate_trace(config)

    def test_session_keeps_gold_target_out_of_agent_observation(self):
        class FakeEnv:
            def __init__(self, **_kwargs):
                pass

            def reset(self, _task_id):
                return {
                    "instruction": "买一双黑色 42 码鞋",
                    "_trace_target": {
                        "asin": "123456789012",
                        "options": {"颜色": "黑色", "尺寸": "42"},
                    },
                }

            def release(self):
                pass

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            state = await session.start(7)
            await session.close()
            return state, session.trace_target

        state, trace_target = asyncio.run(run())
        self.assertNotIn("123456789012", state["latest_observation"])
        self.assertEqual(
            trace_target,
            '最终应购买商品：{"asin":"123456789012","options":{"尺寸":"42","颜色":"黑色"}}',
        )

    def test_target_layout_and_credit_are_deterministic(self):
        self.assertEqual(
            canonical_purchase_target(
                " 123456789012 ", {"颜色": "黑色", "尺寸": "42"}
            ),
            '最终应购买商品：{"asin":"123456789012","options":{"尺寸":"42","颜色":"黑色"}}',
        )

        response_mask = [1, 1, 0, 0, 1, 0, 0, 0]
        response_attention = [1, 1, 1, 1, 1, 1, 0, 0]
        self.assertEqual(
            trace_turn_layout(response_mask, response_attention),
            [((0, 2), 4), ((4, 5), 6)],
        )

        credits = turn_rewards(
            [-2.0, -1.0, -0.5],
            outcome_advantage=1.0,
            epsilon=0.1,
            horizon=1,
            discount=0.8,
            terminal_weight=2.0,
        )
        expected = [
            math.log(2.1 / 1.1),
            math.log(1.1 / 0.6) + 2.0 * 0.8,
        ]
        for actual, wanted in zip(credits, expected, strict=True):
            self.assertAlmostEqual(actual, wanted)

        mixed = mixed_token_advantages(
            outcome_advantage=1.0,
            response_mask=response_mask,
            response_attention=response_attention,
            mean_target_log_probs=[-2.0, -1.0, -0.5],
            epsilon=0.1,
            horizon=1,
            discount=0.8,
            terminal_weight=2.0,
            outcome_weight=1.0,
            turn_weight=0.2,
        )
        self.assertEqual(mixed[2:4], [0.0, 0.0])
        self.assertEqual(mixed[5:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(mixed[0], 1.0 + 0.2 * expected[0])
        self.assertAlmostEqual(mixed[4], 1.0 + 0.2 * expected[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
