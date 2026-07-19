#!/usr/bin/env python3
"""采集一批轨迹，并在同一目录内重建 accepted / rejected / SFT 数据。

raw.jsonl 是断点续跑的唯一事实来源；重复运行同一 output-dir 时，只会补齐缺失
的 task-attempt。每次结束都会从完整 raw 重新生成派生产物，避免增量文件漂移。
"""

import argparse
import json
import os
import signal
from pathlib import Path

from shopping_grpo.sft_data import acceptance_reasons, process_raw_trajectories
from shopping_grpo.teacher_rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    collect_tasks,
    load_tasks,
    rollout_interrupted,
)


def batch_paths(output_dir):
    """返回一个批次的固定文件布局，便于断点续跑和清理。"""
    return {
        "raw": output_dir / "raw.jsonl",
        "accepted": output_dir / "accepted.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "stats": output_dir / "reject_stats.json",
        "sft": output_dir / "sft.jsonl",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Collect one resumable rollout batch and build SFT data.")
    parser.add_argument("--tasks", type=Path, default=Path("data/shop_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collection_100"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--target-accepted",
        type=int,
        default=None,
        help="达到指定 accepted 数即停止；--limit 仍是最多尝试多少个任务。",
    )
    parser.add_argument("--base-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5000"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default="high")
    parser.add_argument("--attempts-per-task", type=int, default=1)
    return parser.parse_args()


def _build_derivatives(paths):
    """从 raw 重建所有派生产物，确保 accepted 与 SFT 使用同一验收规则。"""
    if not paths["raw"].exists():
        return {"total": 0, "accepted": 0, "rejected": 0, "reject_reasons": {}}
    return process_raw_trajectories(
        raw_path=paths["raw"],
        accepted_path=paths["accepted"],
        rejected_path=paths["rejected"],
        stats_path=paths["stats"],
        sft_path=paths["sft"],
    )


def _accepted_count(raw_path):
    """按当前确定性规则统计已有 raw，保证断点续跑不会重复收集。"""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return 0
    with raw_path.open(encoding="utf-8") as handle:
        return sum(
            acceptance_reasons(json.loads(line))[0]
            for line in handle
            if line.strip()
        )


def _progress_bar(total, initial, target_accepted):
    """创建终端进度条；依赖缺失时给出明确的安装命令。"""
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "缺少 tqdm。请先执行：python3 -m pip install -r requirements.txt"
        ) from exc
    return tqdm(
        total=total,
        initial=initial,
        desc="采集任务",
        unit="task",
        dynamic_ncols=True,
        postfix={"accepted": f"{target_accepted} / {target_accepted}"},
    )


def _collect_until_target(
    tasks,
    target_accepted,
    client,
    output_path,
    base_url,
    max_steps,
    attempts_per_task,
):
    """逐任务复用采集器，到目标 accepted 数或候选任务耗尽时停止。"""
    accepted = _accepted_count(output_path)
    written = []
    progress = _progress_bar(
        total=len(tasks), initial=0, target_accepted=target_accepted
    )
    progress.set_postfix_str(f"accepted={accepted}/{target_accepted}")
    try:
        for task in tasks:
            if accepted >= target_accepted:
                break
            new_rows = collect_tasks(
                tasks=[task],
                client=client,
                output_path=output_path,
                base_url=base_url,
                max_steps=max_steps,
                attempts_per_task=attempts_per_task,
            )
            written.extend(new_rows)
            accepted += sum(acceptance_reasons(row)[0] for row in new_rows)
            progress.update(1)
            progress.set_postfix_str(f"accepted={accepted}/{target_accepted}")
    finally:
        progress.close()
    return written, accepted


def main():
    args = parse_args()
    if not args.llm_base_url:
        raise SystemExit("--llm-base-url or OPENAI_BASE_URL is required")
    if not args.api_key:
        raise SystemExit("--api-key or OPENAI_API_KEY is required")
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.target_accepted is not None and args.target_accepted < 1:
        raise SystemExit("--target-accepted must be at least 1")

    # 与底层采集命令保持一致：收到中断后让当前 trajectory 走 finally 并归还租约。
    signal.signal(signal.SIGTERM, rollout_interrupted)
    signal.signal(signal.SIGINT, rollout_interrupted)

    paths = batch_paths(args.output_dir)
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)[: args.limit]
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )

    exit_code = 0
    try:
        if args.target_accepted is None:
            written = collect_tasks(
                tasks=tasks,
                client=client,
                output_path=paths["raw"],
                base_url=args.base_url,
                max_steps=args.max_steps,
                attempts_per_task=args.attempts_per_task,
            )
        else:
            written, accepted = _collect_until_target(
                tasks=tasks,
                target_accepted=args.target_accepted,
                client=client,
                output_path=paths["raw"],
                base_url=args.base_url,
                max_steps=args.max_steps,
                attempts_per_task=args.attempts_per_task,
            )
            print(f"目标 accepted={args.target_accepted}，当前 accepted={accepted}")
        print(f"本次新增 {len(written)} 条 raw trajectory")
    except CollectionInfrastructureError as exc:
        # raw 已经记录到故障前的最后一条；仍生成可检查的派生产物，供下次续跑。
        print(f"采集因基础设施问题暂停：{exc}")
        exit_code = 2

    summary = _build_derivatives(paths)
    print(
        f"batch={args.output_dir} total={summary['total']} "
        f"accepted={summary['accepted']} rejected={summary['rejected']}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
