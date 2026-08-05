#!/usr/bin/env python3
"""Roll out a tool-calling model on manifest-bound frozen Feed test tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

from shopping_grpo.feed.model_rollout import rollout_frozen_dataset


def openai_compatible_completion(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    temperature: float,
):
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def complete(messages, tools):
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "temperature": temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"model endpoint request failed: {exc.__class__.__name__}"
            ) from exc
        try:
            return payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("model endpoint returned no assistant message") from exc

    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    complete = openai_compatible_completion(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env),
        timeout=args.timeout,
        temperature=args.temperature,
    )
    rows = rollout_frozen_dataset(
        args.dataset_dir,
        args.output,
        complete,
        policy_id=args.policy_id,
    )
    print(
        json.dumps(
            {
                "schema_version": "feed-model-rollout-v1",
                "policy_id": args.policy_id,
                "episodes": len(rows),
                "output": str(args.output),
                "training_started": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
