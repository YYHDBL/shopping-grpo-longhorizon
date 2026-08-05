import json
import unittest

from shopping_grpo.feed.model_rollout import rollout_task
from shopping_grpo.feed.observation import render_feed_observation
from shopping_grpo.feed.schema import EpisodeSeed, Persona, Product, Video
from shopping_grpo.feed.simulator import FeedShoppingEnv
from shopping_grpo.feed.tools import FEED_TOOL_SCHEMAS


class FeedModelRolloutTest(unittest.TestCase):
    def test_frozen_model_driver_exposes_only_public_delta_messages(self):
        product = Product("P1", "Item", "storage", 10.0)
        seed = EpisodeSeed(
            "model-eval",
            Persona("persona-model"),
            tuple(
                Video(
                    f"V{index}",
                    caption="Public caption",
                    related_product_ids=("P1",),
                    metadata={"answer_key": "secret"},
                )
                for index in range(24)
            ),
            ("P1",),
            9,
        )
        initial = FeedShoppingEnv(seed, {"P1": product}).observation()
        task = {
            "episode_seed": seed.to_dict(),
            "catalog_products": [product.to_dict()],
            "prompt": [
                {"role": "system", "content": "Use Feed tools."},
                {"role": "user", "content": render_feed_observation(initial)},
            ],
            "tools": FEED_TOOL_SCHEMAS,
        }
        seen = []

        def complete(messages, tools):
            self.assertEqual(tools, FEED_TOOL_SCHEMAS)
            serialized = json.dumps(messages, ensure_ascii=False)
            self.assertNotIn("related_product_ids", serialized)
            self.assertNotIn("answer_key", serialized)
            self.assertNotIn("reward_breakdown", serialized)
            seen.append(messages)
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"commit-{len(seen):03d}",
                        "type": "function",
                        "function": {
                            "name": "commit_recommendation",
                            "arguments": json.dumps(
                                {
                                    "decision": "no_recommend",
                                    "surface": "none",
                                    "strategy": "none",
                                    "relationship": "primary",
                                    "product_ids": [],
                                    "evidence_ids": [],
                                }
                            ),
                        },
                    }
                ],
            }

        result = rollout_task(task, complete, policy_id="model-test")
        self.assertEqual(len(result["transitions"]), 24)
        self.assertTrue(result["transitions"][-1]["done"])
        self.assertEqual(len(seen), 24)
        self.assertTrue(
            any("feed-tool-delta-v1" in message.get("content", "") for message in seen[-1])
        )


if __name__ == "__main__":
    unittest.main()
