#!/usr/bin/env python3
"""从已跑过的 benchmark v1 冻结更小的可比较子集。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.grpo_tasks import freeze_benchmark_subset, read_jsonl, sha256_file, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="冻结既有 ShopSimulator benchmark 的有序子集")
    parser.add_argument("--parent", type=Path, default=Path("data/benchmarks/shop_benchmark_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/shop_benchmark_v2_50.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("data/benchmarks/shop_benchmark_v2_50.metadata.json"))
    parser.add_argument("--size", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    rows, metadata = freeze_benchmark_subset(
        read_jsonl(args.parent), args.parent.stem, args.size, sha256_file(args.parent)
    )
    metadata.update(
        {
            "benchmark_version": args.output.stem,
            "protocol": {"max_steps": 35, "temperature": 0.0, "attempts_per_task": 1, "max_tokens": 512},
        }
    )
    write_jsonl(args.output, rows)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
