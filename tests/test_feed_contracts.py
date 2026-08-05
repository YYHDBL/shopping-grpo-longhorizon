"""Public contracts for the parallel long-horizon feed profile."""

import json
from pathlib import Path
import unittest

from shopping_grpo.feed.actions import action_reject_reason
from shopping_grpo.feed.observation import (
    FEED_OBSERVATION_VERSION,
    FeedObservationError,
    parse_feed_observation,
    render_feed_observation,
)
from shopping_grpo.feed.tools import (
    FEED_TOOL_SCHEMAS,
    MAX_INFO_TOOL_CALLS_PER_VIDEO,
    feed_tool_call_to_action,
    validate_tool_arguments,
)


EXPECTED_TOOLS = [
    "retrieve_products",
    "inspect_product",
    "compare_products",
    "read_reviews",
    "find_alternatives",
    "find_complements",
    "check_inventory",
    "commit_recommendation",
]


def guard_state(**updates):
    state = {
        "visible_product_ids": ["P001", "P002"],
        "evidence_ids": ["video.object.storage_box", "product.P001.price"],
        "purchased_product_ids": [],
        "info_tool_calls": 0,
    }
    state.update(updates)
    return state


def commit(**updates):
    arguments = {
        "decision": "recommend",
        "surface": "product_card",
        "product_ids": ["P001"],
        "relationship": "primary",
        "strategy": "review_summary",
        "evidence_ids": ["video.object.storage_box", "product.P001.price"],
    }
    arguments.update(updates)
    return arguments


def observation_state(**updates):
    state = {
        "observation_version": FEED_OBSERVATION_VERSION,
        "environment_version": "feed-env-v1",
        "episode_id": "episode-1",
        "step": 4,
        "total_steps": 24,
        "persona": {"country": "US", "style_preferences": ["minimal"]},
        "current_video": {
            "video_id": "V004",
            "caption": "Small home-office refresh",
            "objects": ["storage_box"],
        },
        "recent_events": [{"video_id": "V003", "event": "like"}],
        "cart": [],
        "purchased_product_ids": [],
        "visible_product_ids": ["P001", "P002"],
        "evidence_ids": ["video.object.storage_box", "product.P001.price"],
        "info_tool_calls": 1,
        "max_info_tool_calls": MAX_INFO_TOOL_CALLS_PER_VIDEO,
        "done": False,
    }
    state.update(updates)
    return state


class FeedToolSchemaTest(unittest.TestCase):
    def test_canonical_tool_names_and_strict_objects(self):
        self.assertEqual(
            [schema["function"]["name"] for schema in FEED_TOOL_SCHEMAS],
            EXPECTED_TOOLS,
        )
        for schema in FEED_TOOL_SCHEMAS:
            with self.subTest(tool=schema["function"]["name"]):
                self.assertFalse(
                    schema["function"]["parameters"]["additionalProperties"]
                )

    def test_commit_is_flat_when_what_how_and_explanation_is_optional(self):
        commit_schema = FEED_TOOL_SCHEMAS[-1]["function"]
        parameters = commit_schema["parameters"]
        self.assertIn("When--What--How", commit_schema["description"])
        self.assertTrue(
            {"decision", "surface", "product_ids", "relationship", "strategy", "evidence_ids"}
            <= set(parameters["properties"])
        )
        self.assertIn("explanation", parameters["properties"])
        self.assertNotIn("explanation", parameters["required"])
        self.assertEqual(parameters["properties"]["product_ids"]["maxItems"], 2)
        compare = FEED_TOOL_SCHEMAS[2]["function"]["parameters"]
        self.assertEqual(compare["properties"]["product_ids"]["maxItems"], 4)

    def test_verl_config_schemas_are_exactly_canonical(self):
        path = Path(__file__).parents[1] / "configs" / "feed_tools.json"
        configured = json.loads(path.read_text(encoding="utf-8"))["tools"]
        self.assertEqual(
            [item["tool_schema"] for item in configured],
            FEED_TOOL_SCHEMAS,
        )
        self.assertEqual(
            {item["class_name"] for item in configured},
            {"shopping_grpo.feed.verl_adapter.FeedShoppingTool"},
        )

    def test_commit_converts_through_typed_feed_action(self):
        action = feed_tool_call_to_action(
            "commit_recommendation",
            commit(explanation="Grounded in the current video and catalog price."),
        )
        self.assertEqual(action.to_dict(), commit(explanation=action.explanation))

    def test_non_empty_strings_reject_whitespace_only_values(self):
        with self.assertRaisesRegex(ValueError, "schema_blank_string:query"):
            validate_tool_arguments("retrieve_products", {"query": "   "})


class FeedActionGuardTest(unittest.TestCase):
    def test_when_what_how_cross_fields_are_coherent(self):
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                commit(strategy="none"),
                guard_state(),
            ),
            "recommendation_requires_strategy",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                {
                    "decision": "delay",
                    "surface": "none",
                    "product_ids": [],
                    "relationship": "alternative",
                    "strategy": "cheaper_alternative",
                    "evidence_ids": [],
                },
                guard_state(),
            ),
            "non_recommendation_has_strategy",
        )
    def test_rejects_fourth_information_call_for_same_video(self):
        reason = action_reject_reason(
            "inspect_product",
            {"product_id": "P001"},
            guard_state(info_tool_calls=3),
        )
        self.assertEqual(reason, "max_info_tool_calls_exceeded")

    def test_rejects_schema_extra_fields_before_execution(self):
        reason = action_reject_reason(
            "inspect_product",
            {"product_id": "P001", "debug": True},
            guard_state(),
        )
        self.assertEqual(reason, "schema_extra_arguments:debug")

    def test_rejects_non_visible_product_and_evidence(self):
        self.assertEqual(
            action_reject_reason(
                "read_reviews", {"product_id": "P999"}, guard_state()
            ),
            "product_not_visible:P999",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                commit(evidence_ids=["latent.purchase_probability"]),
                guard_state(),
            ),
            "evidence_not_visible:latent.purchase_probability",
        )

    def test_rejects_marketing_after_purchase_but_allows_no_recommend(self):
        purchased = guard_state(purchased_product_ids=["P001"])
        self.assertEqual(
            action_reject_reason("commit_recommendation", commit(), purchased),
            "repeat_marketing_after_purchase",
        )
        no_recommend = commit(
            decision="no_recommend",
            surface="none",
            product_ids=[],
            strategy="none",
            evidence_ids=[],
        )
        self.assertIsNone(
            action_reject_reason(
                "commit_recommendation",
                no_recommend,
                purchased,
            )
        )

    def test_guard_requires_machine_readable_state(self):
        reason = action_reject_reason(
            "inspect_product",
            {"product_id": "P001"},
            'visible_product_ids: ["P001"]',
        )
        self.assertEqual(reason, "invalid_guard_state:not_an_object")

    def test_bundle_requires_two_products_and_evidence_for_each(self):
        bundle_state = guard_state(
            evidence_ids=[
                "video.object.storage_box",
                "product.P001.price",
                "product.P002.rating",
            ]
        )
        two_products = commit(
            product_ids=["P001", "P002"],
            evidence_ids=bundle_state["evidence_ids"],
        )
        self.assertEqual(
            action_reject_reason("commit_recommendation", two_products, bundle_state),
            "multi_product_requires_bundle_relationship",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                commit(relationship="bundle"),
                guard_state(),
            ),
            "bundle_relationship_requires_two_products",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                commit(strategy="bundle"),
                guard_state(),
            ),
            "bundle_strategy_requires_two_products",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                commit(surface="bundle"),
                guard_state(),
            ),
            "bundle_surface_requires_two_products",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                {
                    **two_products,
                    "relationship": "bundle",
                    "evidence_ids": [
                        "video.object.storage_box",
                        "product.P001.price",
                    ],
                },
                bundle_state,
            ),
            "missing_product_evidence:P002",
        )
        self.assertEqual(
            action_reject_reason(
                "commit_recommendation",
                commit(evidence_ids=["product.P001.price"]),
                guard_state(),
            ),
            "missing_context_evidence",
        )
        self.assertIsNone(
            action_reject_reason(
                "commit_recommendation",
                {**two_products, "relationship": "bundle", "strategy": "bundle"},
                bundle_state,
            )
        )


class FeedObservationTest(unittest.TestCase):
    def test_allowlisted_state_round_trips(self):
        rendered = render_feed_observation(observation_state())
        public = parse_feed_observation(rendered)
        self.assertEqual(public["visible_product_ids"], ["P001", "P002"])
        self.assertEqual(public["max_info_tool_calls"], 3)

    def test_unknown_top_level_field_is_rejected(self):
        with self.assertRaisesRegex(FeedObservationError, "non-public fields"):
            render_feed_observation(observation_state(debug_state={"seed": 7}))

    def test_nested_latent_or_probability_field_is_never_rendered(self):
        for field in (
            "latent_state",
            "click_probability",
            "p_click_prob",
            "trust",
            "budget_remaining",
            "price_sensitivity",
            "hindsight_correct",
            "qualified",
        ):
            with self.subTest(field=field):
                state = observation_state()
                state["persona"] = {"country": "US", field: 0.9}
                with self.assertRaisesRegex(FeedObservationError, "sensitive field"):
                    render_feed_observation(state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
