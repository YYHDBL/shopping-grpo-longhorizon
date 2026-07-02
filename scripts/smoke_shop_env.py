#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def build_action_url(base_url, session_id, action):
    if action.startswith("search[") and action.endswith("]"):
        query = action[len("search["):-1]
        keywords = quote(repr([query]))
        return f"{base_url.rstrip('/')}/search_results/{session_id}/{keywords}/1"
    raise ValueError(f"Unsupported smoke action: {action}")


def fetch(url):
    request = Request(url, headers={"User-Agent": "shopping-grpo-smoke/0.1"})
    with urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8", errors="replace")
        return {
            "url": response.geturl(),
            "status_code": response.status,
            "text": body[:2000],
        }


def write_run_result(output_dir, task_id, base_url, steps):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"task_{task_id:04d}_{timestamp}.json"
    payload = {
        "task_id": task_id,
        "base_url": base_url,
        "steps": steps,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def run_smoke(base_url, task_id, actions, output_dir):
    session_id = f"fixed_{task_id}"
    steps = []

    start_url = f"{base_url.rstrip('/')}/{session_id}"
    start_result = fetch(start_url)
    start_result["action"] = "start"
    steps.append(start_result)

    for action in actions:
        action_url = build_action_url(base_url, session_id, action)
        result = fetch(action_url)
        result["action"] = action
        steps.append(result)

    return write_run_result(output_dir, task_id, base_url, steps)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test a ShopSimulator HTTP app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
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
