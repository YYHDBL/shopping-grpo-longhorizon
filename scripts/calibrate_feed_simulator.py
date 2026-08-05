#!/usr/bin/env python3
"""Calibrate observable user-event rates for the numerical Feed simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.feed.calibration import BehaviorCalibration
from shopping_grpo.feed.manifest import sha256_file, write_json_atomic
from shopping_grpo.feed.schema import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="event JSONL or aggregate JSON object")
    parser.add_argument("output", type=Path, help="calibration JSON destination")
    parser.add_argument("--smoothing", type=float, default=10.0)
    return parser.parse_args()


def calibrate_file(
    events_path: str | Path,
    output_path: str | Path,
    *,
    smoothing: float = 10.0,
) -> dict[str, object]:
    source = Path(events_path)
    if source.suffix.lower() == ".json":
        events = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(events, dict):
            raise ValueError("aggregate JSON input must be an object")
    else:
        events = list(iter_jsonl(source))
    calibration = BehaviorCalibration.from_events(events, smoothing=smoothing)
    artifact: dict[str, object] = {
        "schema_version": "feed-behavior-calibration-v1",
        "source": {
            "path": source.name,
            "sha256": sha256_file(source),
        },
        "smoothing": float(smoothing),
        **calibration.to_dict(),
    }
    write_json_atomic(output_path, artifact)
    return artifact


def main() -> None:
    args = parse_args()
    artifact = calibrate_file(args.events, args.output, smoothing=args.smoothing)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
