import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shopping_grpo.feed.calibration import BehaviorCalibration
from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.schema import Product
from shopping_grpo.feed.workflow import run_cpu_mvp, verify_cpu_mvp


def catalog():
    products = []
    for category_index, category in enumerate(("storage", "lamp", "cable", "desk")):
        for index in range(5):
            products.append(
                Product(
                    product_id=f"P{category_index}{index}",
                    title=f"{category} item {index}",
                    category=category,
                    price=float(10 + category_index * 6 + index),
                    attributes=(category, f"style-{index % 2}"),
                    rating=4.0 + index / 10,
                    popularity=float(20 - index),
                    inventory=5,
                )
            )
    return ProductCatalog(products)


class FeedWorkflowTest(unittest.TestCase):
    def test_complete_cpu_chain_produces_hashed_dashboard(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "mvp"
            manifest = run_cpu_mvp(
                catalog(),
                root,
                episodes=3,
                feed_length=24,
                seed=7,
                calibration=BehaviorCalibration.default().to_dict(),
            )
            reopened = json.loads((root / "workflow_manifest.json").read_text(encoding="utf-8"))
            dashboard = root / reopened["artifacts"]["dashboard"]["path"]
            report = json.loads((root / "evaluation" / "report.json").read_text(encoding="utf-8"))
            online_task = json.loads(
                (root / "online_rl_tasks" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            verified = verify_cpu_mvp(root)
            repeated = run_cpu_mvp(
                catalog(),
                root,
                episodes=3,
                feed_length=24,
                seed=7,
                force=True,
                calibration=BehaviorCalibration.default().to_dict(),
            )
            verified_repeated = verify_cpu_mvp(root)
            dashboard_exists = dashboard.is_file()
        self.assertEqual(manifest, reopened)
        self.assertTrue(dashboard_exists)
        self.assertEqual(report["policy_count"], 5)
        self.assertFalse(reopened["training_started"])
        self.assertFalse(reopened["llm_judge"])
        self.assertTrue(reopened["config"]["calibrated"])
        self.assertIn("calibration", online_task)
        self.assertEqual(len(reopened["manifest_content_sha256"]), 64)
        self.assertTrue(verified["ok"])
        self.assertEqual(repeated, manifest)
        self.assertTrue(verified_repeated["ok"])


if __name__ == "__main__":
    unittest.main()
