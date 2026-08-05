import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.calibrate_feed_simulator import calibrate_file
from shopping_grpo.feed.calibration import BehaviorCalibration
from shopping_grpo.feed.simulator import FeedShoppingEnv

from tests.test_feed_simulator import fixture


class FeedCalibrationRuntimeTest(unittest.TestCase):
    def test_conditional_rates_change_runtime_intercepts(self):
        seed, catalog = fixture()
        baseline = FeedShoppingEnv(seed, catalog, calibration=BehaviorCalibration.default())
        high_click = BehaviorCalibration.from_dict(
            {**BehaviorCalibration.default().to_dict(), "click_probability": 0.8}
        )
        calibrated = FeedShoppingEnv(seed, catalog, calibration=high_click)
        self.assertGreater(
            calibrated._coef("click_bias", -2.1),
            baseline._coef("click_bias", -2.1),
        )
        self.assertTrue(math.isfinite(calibrated._coef("click_bias", -2.1)))

    def test_cli_helper_writes_hashed_calibration_artifact(self):
        aggregates = {
            "impression": 100,
            "watch": {"count": 60, "total_dwell_seconds": 600},
            "skip": 40,
            "click": 12,
            "cart": 6,
            "purchase": {"count": 3, "total_value": 240},
            "return": 1,
        }
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "aggregate.json"
            source.write_text(json.dumps(aggregates), encoding="utf-8")
            output = root / "calibration.json"
            artifact = calibrate_file(source, output, smoothing=0.0)
            reopened = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(reopened, artifact)
        self.assertEqual(artifact["click_probability"], 0.12)
        self.assertEqual(artifact["cart_probability"], 0.5)
        self.assertEqual(artifact["purchase_probability"], 0.5)
        self.assertEqual(artifact["return_probability"], 1 / 3)
        self.assertEqual(len(artifact["source"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
