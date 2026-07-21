#!/usr/bin/env python3
"""创建 GRPO probe 候选池，并按 SFT policy 的实测步数冻结训练任务集。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.grpo_tasks import (
    build_grpo_candidate_manifest,
    read_jsonl,
    select_stratified_grpo_tasks,
    sha256_file,
    task_ids,
    write_jsonl,
)


def _candidate_parser(subparsers):
    parser = subparsers.add_parser("candidate", help="生成未见题 probe 候选池")
    parser.add_argument("--tasks", type=Path, default=Path("data/shop_tasks.jsonl"))
    parser.add_argument("--exclude-rollouts", type=Path, default=Path("outputs/flash_accepted_500_parallel/raw.jsonl.gz"))
    parser.add_argument(
        "--exclude-sft",
        type=Path,
        default=None,
        help="可选：在已发布 raw snapshot 之外，又新增且已冻结的 SFT JSONL。",
    )
    parser.add_argument("--exclude-benchmark", type=Path, default=Path("data/benchmarks/shop_benchmark_v2_50.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/grpo_probe_pool_v1.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("data/splits/grpo_probe_pool_v1.metadata.json"))
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)


def _select_parser(subparsers):
    parser = subparsers.add_parser("select", help="从已跑 probe 中分层选择 GRPO 训练 task")
    parser.add_argument("--candidates", type=Path, default=Path("data/splits/grpo_probe_pool_v1.jsonl"))
    parser.add_argument("--probes", type=Path, required=True, help="冻结 SFT policy 在候选池上的 raw rollout JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/splits/grpo_train_v1.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("data/splits/grpo_train_v1.metadata.json"))
    parser.add_argument("--short", type=int, default=300, help="<=10 个执行工具步")
    parser.add_argument("--medium", type=int, default=450, help="11~20 个执行工具步")
    parser.add_argument("--long", type=int, default=250, help=">=21 个执行工具步")
    parser.add_argument("--seed", type=int, default=20260721)


def parse_args():
    parser = argparse.ArgumentParser(description="ShopSimulator GRPO task split")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _candidate_parser(subparsers)
    _select_parser(subparsers)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "candidate":
        task_rows = read_jsonl(args.tasks)
        raw_rows = read_jsonl(args.exclude_rollouts)
        sft_rows = read_jsonl(args.exclude_sft) if args.exclude_sft else []
        benchmark_rows = read_jsonl(args.exclude_benchmark)
        rows = build_grpo_candidate_manifest(
            task_ids(task_rows), task_ids(raw_rows) | task_ids(sft_rows), task_ids(benchmark_rows), args.size, args.seed
        )
        metadata = {
            "split_version": args.output.stem,
            "purpose": "probe SFT policy rollout lengths before GRPO stratification",
            "task_count": len(rows),
            "seed": args.seed,
            "source_tasks": str(args.tasks),
            "source_tasks_sha256": sha256_file(args.tasks),
            "excluded_raw_rollouts": str(args.exclude_rollouts),
            "excluded_raw_rollouts_sha256": sha256_file(args.exclude_rollouts),
            "excluded_raw_rollout_task_count": len(task_ids(raw_rows)),
            "extra_excluded_sft": str(args.exclude_sft) if args.exclude_sft else None,
            "extra_excluded_sft_sha256": sha256_file(args.exclude_sft) if args.exclude_sft else None,
            "extra_excluded_sft_task_count": len(task_ids(sft_rows)),
            "excluded_benchmark": str(args.exclude_benchmark),
            "excluded_benchmark_sha256": sha256_file(args.exclude_benchmark),
            "excluded_benchmark_task_count": len(task_ids(benchmark_rows)),
            "selection": "deterministic_random_without_replacement",
        }
    else:
        rows, report = select_stratified_grpo_tasks(
            read_jsonl(args.candidates),
            read_jsonl(args.probes),
            {"short": args.short, "medium": args.medium, "long": args.long},
            args.seed,
        )
        metadata = {
            "split_version": args.output.stem,
            "purpose": "online Vanilla GRPO task ids; no teacher trajectory is used as training target",
            "seed": args.seed,
            "candidate_pool": str(args.candidates),
            "candidate_pool_sha256": sha256_file(args.candidates),
            "probe_rollouts": str(args.probes),
            "probe_rollouts_sha256": sha256_file(args.probes),
            "length_definition": {"short": "<=10 executed tool steps", "medium": "11-20", "long": ">=21"},
            **report,
        }
    write_jsonl(args.output, rows)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
