#!/usr/bin/env python3
"""Generate the five reproducible long-horizon Feed dataset artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.feed.calibration import BehaviorCalibration
from shopping_grpo.feed.datasets import (
    MAX_FEED_LENGTH,
    MIN_FEED_LENGTH,
    generate_feed_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    REPOSITORY_ROOT
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "data"
    / "fine_items_eval_train_all.json.gz"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Feed JSONL or ShopSimulator JSON/JSON.GZ product archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for artifacts and manifest.json.",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--feed-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="optional feed-behavior-calibration-v1 JSON produced by calibrate_feed_simulator.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only known generated artifacts in a non-empty output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 3:
        raise SystemExit("--episodes must be at least 3")
    if not MIN_FEED_LENGTH <= args.feed_length <= MAX_FEED_LENGTH:
        raise SystemExit(
            f"--feed-length must be from {MIN_FEED_LENGTH} to {MAX_FEED_LENGTH}"
        )
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    calibration = None
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        BehaviorCalibration.from_dict(calibration)
    manifest = generate_feed_artifacts(
        args.catalog,
        args.output_dir,
        episodes=args.episodes,
        feed_length=args.feed_length,
        seed=args.seed,
        force=args.force,
        calibration=calibration,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "artifact_counts": manifest["artifact_counts"],
                "curriculum_stage_counts": manifest["curriculum_stage_counts"],
                "manifest_content_sha256": manifest["manifest_content_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
