#!/usr/bin/env python3
"""从 ShopSimulator task 清单中创建固定的 held-out benchmark。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.benchmark import build_benchmark_manifest


def _task_ids(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [int(json.loads(line)["task_id"]) for line in handle if line.strip()]


def create_benchmark(tasks_path, sft_path, manifest_path, metadata_path, size, seed):
    """创建 benchmark 文件，并记录其与当前 SFT 快照的隔离关系。"""
    all_task_ids = _task_ids(tasks_path)
    sft_task_ids = set(_task_ids(sft_path))
    rows = build_benchmark_manifest(all_task_ids, sft_task_ids, size=size, seed=seed)
    manifest_path = Path(manifest_path)
    metadata_path = Path(metadata_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    metadata = {
        "benchmark_version": manifest_path.stem,
        "task_count": len(rows),
        "seed": int(seed),
        "source_task_count": len(set(all_task_ids)),
        "excluded_sft_task_count": len(sft_task_ids),
        "protocol": {"max_steps": 35, "temperature": 0.0, "attempts_per_task": 1},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="创建与 SFT task 隔离的 ShopSimulator benchmark")
    parser.add_argument("--tasks", type=Path, default=Path("data/shop_tasks.jsonl"))
    parser.add_argument(
        "--sft", type=Path, default=Path("outputs/flash_accepted_500_parallel/sft.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/shop_benchmark_v1.jsonl")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/benchmarks/shop_benchmark_v1.metadata.json")
    )
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = create_benchmark(
        args.tasks, args.sft, args.output, args.metadata, size=args.size, seed=args.seed
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
