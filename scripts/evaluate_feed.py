#!/usr/bin/env python3
"""Build a paired, deterministic report from frozen Feed policy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.feed.report import evaluate_log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-dir", type=Path, help="verify its manifest before scoring")
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="sealed checkpoint/run identity for model-produced rollout logs",
    )
    parser.add_argument("--allow-unpaired", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_log_file(
        args.logs,
        args.output_dir,
        split=args.split,
        dataset_dir=args.dataset_dir,
        run_manifest=args.run_manifest,
        require_paired=not args.allow_unpaired,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "split": report["split"],
                "policy_count": report["policy_count"],
                "episode_count_per_policy": report["episode_count_per_policy"],
                "report": str(args.output_dir / "report.json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
