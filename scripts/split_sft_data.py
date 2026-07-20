#!/usr/bin/env python3
"""按 task_id 将 SFT JSONL 稳定划分为训练集与验证集。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.sft_training import split_rows_by_task


def parse_args():
    parser = argparse.ArgumentParser(description="按 task_id 划分 SFT 训练集和验证集")
    parser.add_argument("--input", type=Path, required=True, help="验收后生成的 sft.jsonl")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows, validation_rows = split_rows_by_task(
        rows, validation_ratio=args.validation_ratio, seed=args.seed
    )
    write_jsonl(args.train, train_rows)
    write_jsonl(args.validation, validation_rows)
    print(
        f"total={len(rows)} train={len(train_rows)} validation={len(validation_rows)} "
        f"train_tasks={len({row.get('task_id') for row in train_rows})} "
        f"validation_tasks={len({row.get('task_id') for row in validation_rows})}"
    )


if __name__ == "__main__":
    main()
