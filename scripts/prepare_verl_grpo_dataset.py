#!/usr/bin/env python3
"""把冻结的 GRPO task_id 转为 veRL 所需 parquet prompt 数据。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shopping_grpo.grpo_tasks import read_jsonl
from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.teacher_rollout import SYSTEM_PROMPT


def build_verl_record(task_id: int, user_instruction: str, split: str, index: int) -> dict:
    """只写模型本应看到的 system、用户需求和 task_id；不写 goal/reward。"""
    return {
        "data_source": "shopsimulator",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_instruction)},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "split": split,
            "index": int(index),
            "task_id": int(task_id),
            "interaction_kwargs": {"name": "shopsimulator", "task_id": int(task_id)},
        },
    }


def fetch_user_instruction(task_id: int, base_url: str, timeout: int) -> str:
    """通过 reset 获取真实用户 query，并在 finally 中归还 probe 租约。"""
    with ShopAgentEnv(base_url=base_url, timeout=timeout) as env:
        initial = env.reset(task_id)
        return str(initial.get("instruction", initial.get("observation", "")))


def parse_args():
    parser = argparse.ArgumentParser(description="生成 veRL ShopSimulator GRPO parquet 数据集")
    parser.add_argument("--tasks", type=Path, required=True, help="grpo_train_v1.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="输出 parquet，例如 data/verl/grpo_train_v1.parquet")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("缺少 pyarrow；请在 veRL 环境执行：uv pip install pyarrow") from exc
    rows = []
    for index, task in enumerate(read_jsonl(args.tasks)):
        task_id = int(task["task_id"])
        instruction = fetch_user_instruction(task_id, args.base_url, args.timeout)
        if not instruction:
            raise SystemExit(f"task_id={task_id} reset 没有返回用户需求，停止生成数据集")
        rows.append(build_verl_record(task_id, instruction, args.split, index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"veRL parquet 已写入 {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
