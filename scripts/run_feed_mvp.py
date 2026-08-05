#!/usr/bin/env python3
"""Run the complete CPU Feed MVP without loading a model or starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.feed.calibration import BehaviorCalibration
from shopping_grpo.feed.datasets import MAX_FEED_LENGTH, MIN_FEED_LENGTH
from shopping_grpo.feed.workflow import run_cpu_mvp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    REPOSITORY_ROOT
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "data"
    / "fine_items_eval_train_all.json.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--feed-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="optional calibration JSON produced by calibrate_feed_simulator.py",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MIN_FEED_LENGTH <= args.feed_length <= MAX_FEED_LENGTH:
        raise SystemExit(
            f"--feed-length must be from {MIN_FEED_LENGTH} to {MAX_FEED_LENGTH}"
        )
    calibration = None
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        BehaviorCalibration.from_dict(calibration)
    manifest = run_cpu_mvp(
        args.catalog,
        args.output_dir,
        episodes=args.episodes,
        feed_length=args.feed_length,
        seed=args.seed,
        force=args.force,
        calibration=calibration,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
