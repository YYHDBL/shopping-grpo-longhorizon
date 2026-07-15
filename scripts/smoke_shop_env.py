#!/usr/bin/env python3
"""Run raw ShopSimulator actions through its structured API."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from shopping_grpo.shop_http_env import ShopAgentEnv


def write_run_result(output_dir, task_id, base_url, steps, reset=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"task_{task_id:04d}_{timestamp}.json"
    payload = {
        "task_id": task_id,
        "base_url": base_url,
        "reset": reset or {},
        "steps": steps,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def run_smoke(base_url, task_id, actions, output_dir, env_factory=ShopAgentEnv):
    steps = []
    with env_factory(base_url=base_url) as env:
        reset = env.reset(task_id)
        for action in actions:
            result = env.step(action)
            steps.append({"action": action, "result": result})
            if result.get("done", False):
                break
    return write_run_result(output_dir, task_id, base_url, steps, reset=reset)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test ShopSimulator structured API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--actions", nargs="+", default=["search[乳胶枕]"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    return parser.parse_args()


def main():
    args = parse_args()
    path = run_smoke(args.base_url, args.task_id, args.actions, args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
