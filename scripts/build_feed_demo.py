#!/usr/bin/env python3
"""Build a standalone HTML evaluator from Feed mixed-policy logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from shopping_grpo.feed.demo import write_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, help="mixed-policy trajectory JSONL")
    parser.add_argument("output", type=Path, help="destination .html")
    parser.add_argument("--summary", type=Path, help="optional evaluation summary JSON")
    parser.add_argument("--title", default="Feed Agent Lab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = write_demo(
        args.logs,
        args.output,
        evaluation_summary_path=args.summary,
        title=args.title,
    )
    print(destination)


if __name__ == "__main__":
    main()
