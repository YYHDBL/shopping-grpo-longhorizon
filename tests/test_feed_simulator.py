from types import SimpleNamespace
import json
import unittest

from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.schema import EpisodeSeed, Persona, Product, Video
from shopping_grpo.feed.simulator import (
    FeedActionGuardError,
    FeedShoppingEnv,
)


class TinyCatalog:
    def __init__(self, products):
        self.products = {product.product_id: product for product in products}

    def get(self, product_id):
        return self.products.get(product_id)

    def search(self, query, **kwargs):
        del query
        candidate_ids = kwargs.get("candidate_ids")
        rows = list(self.products.values())
        if candidate_ids:
            rows = [row for row in rows if row.product_id in set(candidate_ids)]
        return rows[: kwargs.get("limit", 8)]

    def alternatives(self, product_id, limit=5):
        category = self.products[product_id].category
        return [
            row
            for row in self.products.values()
            if row.product_id != product_id and row.category == category
        ][:limit]

    def complements(self, product_id, limit=5):
        ids = set(self.products[product_id].complement_product_ids)
        return [row for row in self.products.values() if row.product_id in ids][:limit]


def fixture():
    products = [
        Product(
            product_id="P1",
            title="Minimal storage box",
            category="storage",
            price=25.0,
            attributes=("storage", "minimal"),
            rating=4.7,
            inventory=10,
            tags=("scratch-risk",),
            complement_product_ids=("P3",),
        ),
        Product(
            product_id="P2",
            title="Budget storage box",
            category="storage",
            price=15.0,
            attributes=("storage",),
            rating=4.2,
            inventory=10,
        ),
        Product(
            product_id="P3",
            title="Cable clips",
            category="cable",
            price=8.0,
            attributes=("cable",),
            rating=4.5,
            inventory=10,
        ),
    ]
    videos = tuple(
        Video(
            video_id=f"V{index}",
            caption="Small home-office makeover",
            scene=("home_office",),
            objects=("storage", "cable"),
            style=("minimal",),
            related_product_ids=("P1", "P2"),
            duration_seconds=30.0,
        )
        for index in range(4)
    )
    seed = EpisodeSeed(
        episode_id="episode-1",
        persona=Persona(
            persona_id="U1",
            budget=100.0,
            category_interests=("storage",),
            style_preferences=("minimal",),
            price_sensitivity=0.6,
        ),
        videos=videos,
        product_ids=("P1", "P2", "P3"),
        seed=17,
        inventory={"P1": 10, "P2": 10, "P3": 10},
    )
    return seed, TinyCatalog(products)


def grounded_action(env, *, surface="review_summary", strategy="review_summary"):
    retrieved = env.call_tool("retrieve_products", {"query": "storage"})
    product_id = retrieved["products"][0]["product_id"]
    inspected = env.call_tool("inspect_product", {"product_id": product_id})
    reviews = env.call_tool("read_reviews", {"product_id": product_id})
    context_evidence = next(
        evidence_id for evidence_id in env.observation()["evidence_ids"] if evidence_id.startswith("video.")
    )
    return {
        "decision": "recommend",
        "surface": surface,
        "strategy": strategy,
        "relationship": "primary",
        "product_ids": [product_id],
        "evidence_ids": [context_evidence, inspected["evidence_ids"][0], reviews["evidence_ids"][0]],
        "explanation": "The item matches the current workspace video; review risk is disclosed.",
    }


class FeedSimulatorTest(unittest.TestCase):
    def test_public_payload_bounds_catalog_output_and_dotted_product_ids(self):
        products = [
            Product(
                product_id="P.1" if index == 0 else f"P{index}",
                title="title-" + "x" * 1_000,
                category="storage",
                price=float(index + 1),
                attributes=tuple("a" * 500 for _ in range(40)),
            )
            for index in range(20)
        ]

        class UnboundedCatalog(TinyCatalog):
            def search(self, query, **kwargs):
                del query, kwargs
                return list(self.products.values())

            def alternatives(self, product_id, limit=5):
                del product_id, limit
                return list(self.products.values())

        video = Video(
            "V1",
            caption="c" * 5_000,
            objects=tuple(f"secret-{index}-" + "o" * 10_000 for index in range(100)),
            style=tuple(f"style-{index}-" + "s" * 10_000 for index in range(100)),
        )
        seed = EpisodeSeed(
            "bounded",
            Persona("U"),
            (video,),
            tuple(product.product_id for product in products),
            3,
        )
        env = FeedShoppingEnv(seed, UnboundedCatalog(products))
        observation = env.observation()
        self.assertLess(len(json.dumps(observation)), 20_000)
        self.assertEqual(len(observation["current_video"]["objects"]), 32)
        self.assertFalse(any("secret-" in item for item in observation["evidence_ids"]))

        retrieved = env.call_tool("retrieve_products", {"query": "storage"})
        self.assertEqual(len(retrieved["products"]), 5)
        dotted_evidence = retrieved["evidence_ids"][0]
        self.assertIn("P%2E1", dotted_evidence)
        alternatives = env.call_tool("find_alternatives", {"product_id": "P.1"})
        self.assertEqual(len(alternatives["alternatives"]), 4)
        context = next(
            item for item in env.observation()["evidence_ids"] if item.startswith("video.")
        )
        result = env.step(
            {
                "decision": "recommend",
                "surface": "product_card",
                "strategy": "direct",
                "relationship": "primary",
                "product_ids": ["P.1"],
                "evidence_ids": [context, dotted_evidence],
            }
        )
        self.assertTrue(result.done)

    def test_public_observation_never_contains_latent_state(self):
        seed, catalog = fixture()
        env = FeedShoppingEnv(seed, catalog)
        observation = env.observation()
        payload = repr(observation).lower()
        for secret in ("latent", "purchase_intent", "trust", "fatigue", "budget_remaining", "probability"):
            self.assertNotIn(secret, payload)
        self.assertNotIn("related_product_ids", observation["current_video"])
        self.assertNotIn("embedding", observation["current_video"])
        self.assertNotIn("metadata", observation["current_video"])

    def test_information_call_limit_and_grounded_guard(self):
        seed, catalog = fixture()
        env = FeedShoppingEnv(seed, catalog)
        action = grounded_action(env)
        with self.assertRaisesRegex(FeedActionGuardError, "max_info_tool_calls"):
            env.call_tool("check_inventory", {"product_ids": ["P1"]})

        result = env.step(action)
        self.assertFalse(result.done)
        self.assertEqual(len(result.info["tool_records"]), 3)

        other = FeedShoppingEnv(seed, catalog)
        with self.assertRaisesRegex(FeedActionGuardError, "product_not_visible"):
            other.step(
                {
                    "decision": "recommend",
                    "surface": "product_card",
                    "strategy": "direct",
                    "product_ids": ["P1"],
                    "evidence_ids": ["video.missing"],
                }
            )

    def test_bundle_is_two_grounded_products_or_is_rejected(self):
        seed, catalog = fixture()
        env = FeedShoppingEnv(seed, catalog)
        retrieved = env.call_tool("retrieve_products", {"query": "storage"})
        product_evidence = {
            row["product_id"]: evidence_id
            for row, evidence_id in zip(
                retrieved["products"], retrieved["evidence_ids"], strict=True
            )
        }
        context_evidence = next(
            evidence_id
            for evidence_id in env.observation()["evidence_ids"]
            if evidence_id.startswith("video.")
        )
        bundle = {
            "decision": "recommend",
            "surface": "bundle",
            "strategy": "bundle",
            "relationship": "bundle",
            "product_ids": ["P1", "P2"],
            "evidence_ids": [
                context_evidence,
                product_evidence["P1"],
                product_evidence["P2"],
            ],
        }

        with self.assertRaisesRegex(
            FeedActionGuardError, "multi_product_requires_bundle_relationship"
        ):
            env.step({**bundle, "relationship": "primary"})
        with self.assertRaisesRegex(FeedActionGuardError, "missing_product_evidence:P2"):
            env.step({**bundle, "evidence_ids": bundle["evidence_ids"][:-1]})

        result = env.step(bundle)
        self.assertFalse(result.done)

    def test_catalog_inferred_complement_is_rewarded_consistently_with_tool(self):
        products = (
            Product("A", "Desk shelf", "storage", 20.0, attributes=("workspace",)),
            Product("B", "Cable clips", "cable", 8.0, attributes=("workspace",)),
        )
        catalog = ProductCatalog(products)
        seed = EpisodeSeed(
            "inferred-complement",
            Persona("U", budget=100.0, category_interests=("storage",)),
            (
                Video(
                    "V",
                    caption="Workspace refresh",
                    objects=("workspace",),
                    related_product_ids=("A",),
                ),
            ),
            ("A", "B"),
            5,
            inventory={"A": 2, "B": 2},
        )
        env = FeedShoppingEnv(seed, catalog)
        retrieved = env.call_tool("retrieve_products", {"query": "workspace"})
        related = env.call_tool("find_complements", {"product_id": "A"})
        self.assertEqual(related["complements"][0]["product_id"], "B")
        context = next(
            item for item in env.observation()["evidence_ids"] if item.startswith("video.")
        )
        evidence = [context, *retrieved["evidence_ids"], *related["evidence_ids"]]
        result = env.step(
            {
                "decision": "recommend",
                "surface": "bundle",
                "strategy": "bundle",
                "relationship": "bundle",
                "product_ids": ["A", "B"],
                "evidence_ids": list(dict.fromkeys(evidence)),
            }
        )
        self.assertGreater(result.reward_breakdown["bundle_value"], 0.0)
        self.assertTrue(
            any(event["event_type"] == "bundle_offer" for event in result.events)
        )

    def test_purchase_is_delayed_and_does_not_end_feed(self):
        seed, catalog = fixture()
        calibration = SimpleNamespace(
            coefficients={"click_bias": 12.0, "cart_bias": 12.0, "buy_bias": 12.0}
        )
        env = FeedShoppingEnv(seed, catalog, calibration=calibration)
        first = env.step(grounded_action(env))
        self.assertFalse(first.done)
        self.assertFalse(any(event["event_type"] == "purchase" for event in first.events))

        while not env.done:
            env.step(
                {
                    "decision": "no_recommend",
                    "surface": "none",
                    "strategy": "none",
                    "product_ids": [],
                    "evidence_ids": [],
                }
            )
        purchases = [
            event
            for transition in env.transitions
            for event in transition["events"]
            if event["event_type"] == "purchase"
        ]
        self.assertTrue(purchases)
        self.assertEqual(purchases[0]["source_step"], 0)
        self.assertGreater(purchases[0]["metadata"]["realized_at_step"], 0)

    def test_common_random_numbers_keep_content_feedback_equal_across_actions(self):
        seed, catalog = fixture()
        factual = FeedShoppingEnv(seed, catalog)
        counterfactual = FeedShoppingEnv(seed, catalog)
        factual_result = factual.step(grounded_action(factual))
        counterfactual_result = counterfactual.step(
            {
                "decision": "no_recommend",
                "surface": "none",
                "strategy": "none",
                "product_ids": [],
                "evidence_ids": [],
            }
        )
        content_types = {"watch", "skip", "like"}
        factual_content = [
            (event["event_type"], event["value"])
            for event in factual_result.events
            if event["event_type"] in content_types
        ]
        counterfactual_content = [
            (event["event_type"], event["value"])
            for event in counterfactual_result.events
            if event["event_type"] in content_types
        ]
        self.assertEqual(factual_content, counterfactual_content)

    def test_reward_total_equals_component_sum(self):
        seed, catalog = fixture()
        env = FeedShoppingEnv(seed, catalog)
        result = env.step(
            {
                "decision": "no_recommend",
                "surface": "none",
                "strategy": "none",
                "product_ids": [],
                "evidence_ids": [],
            }
        )
        components = sum(
            value for key, value in result.reward_breakdown.items() if key != "total"
        )
        self.assertAlmostEqual(result.reward, components)

    def test_session_break_is_observable_but_preserves_episode(self):
        seed, catalog = fixture()
        seed = EpisodeSeed.from_dict({**seed.to_dict(), "session_breaks": [1]})
        env = FeedShoppingEnv(seed, catalog)
        env.step(
            {
                "decision": "no_recommend",
                "surface": "none",
                "strategy": "none",
                "product_ids": [],
                "evidence_ids": [],
            }
        )
        result = env.step(
            {
                "decision": "delay",
                "surface": "none",
                "strategy": "none",
                "product_ids": [],
                "evidence_ids": [],
            }
        )
        self.assertFalse(result.done)
        self.assertIn("session_break", {event["event_type"] for event in result.events})


if __name__ == "__main__":
    unittest.main()
