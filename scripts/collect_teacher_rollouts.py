#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from shopping_grpo.teacher_rollout import OpenAIChatClient, collect_tasks, load_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Collect raw teacher rollouts with OpenAI tool calls.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/rollouts/teacher_raw.jsonl"))
    parser.add_argument("--base-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5000"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.llm_base_url:
        raise SystemExit("--llm-base-url or OPENAI_BASE_URL is required")
    if not args.api_key:
        raise SystemExit("--api-key or OPENAI_API_KEY is required")

    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
    )
    written = collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        attempts_per_task=args.attempts_per_task,
    )
    print(f"wrote {len(written)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
