import gzip
import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.feed import (
    BehaviorCalibration,
    Decision,
    EpisodeResult,
    EpisodeSeed,
    EventType,
    FeedAction,
    FeedTransition,
    Persona,
    Product,
    ProductCatalog,
    Relationship,
    RewardBreakdown,
    Strategy,
    Surface,
    ToolRecord,
    UserEvent,
    Video,
    load_jsonl,
    write_jsonl,
)


def _products():
    return (
        Product(
            "p1",
            "Red desktop storage box",
            "Home›Storage",
            20.0,
            attributes=("minimal", "desktop"),
            tags=("scratch-risk",),
            complement_product_ids=("p2",),
            embedding=(1.0, 0.0),
        ),
        Product(
            "p2",
            "Warm desk lamp",
            "Home›Lighting",
            30.0,
            attributes=("warm", "desktop"),
            embedding=(0.0, 1.0),
        ),
        Product(
            "p3",
            "Blue storage organizer",
            "Home›Storage",
            15.0,
            attributes=("minimal",),
            embedding=(0.8, 0.2),
        ),
    )


class FeedFoundationTests(unittest.TestCase):
    def test_schema_round_trip_and_canonical_flat_action(self):
        product = _products()[0]
        self.assertEqual(Product.from_dict(product.to_dict()), product)
        self.assertEqual(product.asin, "p1")

        video = Video(
            "v1",
            caption="Small home-office makeover",
            scene=("home_office",),
            objects=("storage_box", "lamp"),
            style=("minimal",),
            related_product_ids=("p1", "p2"),
        )
        persona = Persona(
            "u1",
            country="CN",
            budget=50.0,
            category_interests=("Storage",),
            style_preferences=("minimal",),
            price_sensitivity=0.7,
        )
        seed = EpisodeSeed(
            "ep1",
            persona,
            (video,),
            ("p1", "p2"),
            7,
            inventory={"p2": 4, "p1": 3},
        )
        self.assertEqual(EpisodeSeed.from_dict(seed.to_dict()), seed)
        self.assertEqual(seed.to_dict()["inventory"], {"p1": 3, "p2": 4})

        action = FeedAction.from_dict(
            {
                "decision": "recommend",
                "surface": "product_card",
                "strategy": "review_summary",
                "relationship": "primary",
                "product_ids": ["p1"],
                "evidence_ids": ["video.v1.objects", "product.p1.price"],
                "explanation": "Matches the observed desk-storage scene.",
            }
        )
        self.assertIs(action.decision, Decision.RECOMMEND)
        self.assertIs(action.surface, Surface.PRODUCT_CARD)
        self.assertIs(action.strategy, Strategy.REVIEW_SUMMARY)
        self.assertIs(action.relationship, Relationship.PRIMARY)
        self.assertEqual(action.reason, action.explanation)
        self.assertNotIn("reason", action.to_dict())
        self.assertEqual(
            FeedAction.from_dict({"decision": "delay", "reason": "observe more"}).explanation,
            "observe more",
        )
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            FeedAction.from_dict({"decision": "delay", "surprise": True})
        with self.assertRaisesRegex(ValueError, "invalid decision"):
            FeedAction.from_dict({"decision": "always_sell"})

        event = UserEvent(EventType.CLICK, step=0, video_id="v1", product_id="p1")
        tool = ToolRecord(
            "retrieve_products",
            {"query": "storage"},
            {"product_ids": ["p1"]},
            ("product.p1.title",),
        )
        reward = RewardBreakdown(engagement=0.1, interruption_penalty=-0.02)
        transition = FeedTransition(
            0,
            "v1",
            action,
            events=(event,),
            reward=reward,
            tool_records=(tool,),
            done=True,
        )
        result = EpisodeResult("ep1", (transition,))
        restored = EpisodeResult.from_dict(result.to_dict())
        self.assertEqual(restored, result)
        self.assertAlmostEqual(restored.total_reward, 0.08)
        self.assertEqual(restored.events, (event,))

    def test_jsonl_is_unicode_safe_and_deterministic(self):
        rows = [Video("v2", caption="收纳"), Video("v1", caption="台灯")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "videos.jsonl.gz"
            write_jsonl(path, rows)
            self.assertEqual(load_jsonl(path, Video), rows)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                lines = handle.readlines()
        self.assertIn("收纳", lines[0])
        self.assertTrue(lines[0].startswith('{"asr":'))

    def test_catalog_lexical_hybrid_relationships_and_jsonl(self):
        catalog = ProductCatalog(reversed(_products()))
        self.assertEqual(catalog.product_ids, ("p1", "p2", "p3"))
        self.assertIsNone(catalog.get("missing"))
        self.assertEqual(
            catalog.search("storage", top_k=2),
            [catalog.require("p3"), catalog.require("p1")],
        )
        self.assertEqual(
            catalog.search("storage", limit=1, candidate_ids=("p1",)),
            [catalog.require("p1")],
        )

        hybrid = catalog.search(
            "lamp",
            query_embedding=(1.0, 0.0),
            lexical_weight=0.1,
            embedding_weight=0.9,
            top_k=1,
        )
        self.assertEqual(hybrid[0].product_id, "p1")
        self.assertEqual(
            [item.product_id for item in catalog.alternatives("p1", cheaper_only=True)], ["p3"]
        )
        self.assertEqual(len(catalog.alternatives("p1", limit=1)), 1)
        self.assertEqual([item.product_id for item in catalog.complements("p1")], ["p2"])
        self.assertEqual([item.product_id for item in catalog.complements("p1", limit=1)], ["p2"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            catalog.to_jsonl(path)
            restored = ProductCatalog.from_jsonl(path)
        self.assertEqual(restored.products, catalog.products)

    def test_shopsimulator_gzip_adapter_uses_product_truth_only(self):
        raw = [
            {
                "asin": "12345678",
                "title": "天然乳胶枕",
                "category": "家居›枕头",
                "pricing": [80, 120],
                "attribute": ["天然", "护颈"],
                "images": ["https://example.test/image.jpg"],
                "customization_options": {"颜色": [{"value": "白色", "price": 80}]},
                "instructions": [{"instruction": "hidden target"}],
                "user_persona": {"用户ID": "hidden user"},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(raw, handle, ensure_ascii=False)
            catalog = ProductCatalog.from_shopsimulator(path, expected_count=1)
        product = catalog.require("12345678")
        self.assertEqual(product.price, 80.0)
        self.assertEqual(product.max_price, 120.0)
        self.assertEqual(product.attributes, ("天然", "护颈"))
        payload = json.dumps(product.to_dict(), ensure_ascii=False)
        self.assertNotIn("hidden target", payload)
        self.assertNotIn("hidden user", payload)
        self.assertEqual(product.metadata["customization_options"]["颜色"][0]["value"], "白色")

    def test_embedded_shopsimulator_archive_has_23421_products(self):
        root = Path(__file__).resolve().parents[1]
        archive = (
            root
            / "environments"
            / "ShopSimulator"
            / "shop_env"
            / "data"
            / "fine_items_eval_train_all.json.gz"
        )
        catalog = ProductCatalog.from_shopsimulator(archive, expected_count=23_421)
        self.assertEqual(len(catalog), 23_421)
        self.assertTrue(all(product.source == "shopsimulator" for product in catalog.products))

    def test_behavior_calibration_uses_aggregate_conditional_rates(self):
        calibration = BehaviorCalibration.from_events(
            {
                "impression": 100,
                "watch": {"count": 60, "mean_dwell_seconds": 12.0},
                "skip": 40,
                "like": 6,
                "share": 3,
                "click": 10,
                "add_to_cart": 5,
                "purchase": {"count": 2, "mean_value": 40.0},
                "return": 1,
            }
        )
        self.assertAlmostEqual(calibration.watch_rate, 0.60)
        self.assertAlmostEqual(calibration.skip_rate, 0.40)
        self.assertAlmostEqual(calibration.like_rate, 0.10)
        self.assertAlmostEqual(calibration.share_rate, 0.05)
        self.assertAlmostEqual(calibration.click_rate, 0.10)
        self.assertAlmostEqual(calibration.cart_rate, 0.50)
        self.assertAlmostEqual(calibration.purchase_rate, 0.40)
        self.assertAlmostEqual(calibration.return_rate, 0.50)
        self.assertAlmostEqual(calibration.mean_dwell_seconds, 12.0)
        self.assertAlmostEqual(calibration.mean_purchase_value, 40.0)
        self.assertEqual(calibration.event_count("exposure"), 100)
        self.assertEqual(BehaviorCalibration.from_dict(calibration.to_dict()), calibration)

        smoothed = BehaviorCalibration.from_events({"impression": 1, "click": 1}, smoothing=1.0)
        self.assertAlmostEqual(smoothed.click_probability, (1.0 + 0.05) / 2.0)


if __name__ == "__main__":
    unittest.main()
