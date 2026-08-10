#!/usr/bin/env python3
"""Label cleaned SFT trajectories with DeepSeek-v4 difficulty judgments."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shopping_grpo.evaluation.model_client import client_from_environment


SYSTEM_PROMPT = """You curate training data for a shopping web agent.
Judge intrinsic TASK difficulty separately from the observed teacher TRAJECTORY complexity.
The number of written constraints alone is weak evidence. Give more weight to product category,
price, variant/specification ambiguity, catalog ambiguity, budget boundary, need to compare
candidates, and need to rewrite a search query. A long or repetitive teacher trajectory may be a
teacher failure mode; do not automatically call the task hard for that reason.

Return one strict JSON object: {"labels": [...]}. Return exactly one label for every task_id.
Each label must contain:
- task_id: integer
- difficulty: "simple", "medium", or "hard" (intrinsic task difficulty)
- trajectory_complexity: "simple", "medium", or "hard"
- domain: short Chinese product-category name
- price_band: "low", "medium", "high", or "unknown"
- needs_query_rewrite: boolean
- needs_candidate_comparison: boolean
- variant_complexity: "low", "medium", or "high"
- budget_risk: "low", "medium", or "high"
- loop_risk: "low", "medium", or "high"
- confidence: number from 0 to 1
- evidence: concise Chinese explanation, at most 80 Chinese characters

Difficulty guide: simple means direct low-ambiguity purchase; medium means some search refinement,
comparison, or specification checking; hard means substantial category/variant ambiguity, expensive
or budget-sensitive purchase, or multi-candidate reasoning likely required.
"""

LEVELS = {"simple", "medium", "hard"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_batches(
    rows: list[dict],
    max_rows: int,
    max_chars: int,
    max_tool_content_chars: int | None = None,
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for row in rows:
        trajectory = [dict(m) for m in row["messages"] if m.get("role") != "system"]
        if max_tool_content_chars:
            for message in trajectory:
                content = message.get("content")
                if (
                    message.get("role") == "tool"
                    and isinstance(content, str)
                    and len(content) > max_tool_content_chars
                ):
                    head = max_tool_content_chars * 3 // 4
                    tail = max_tool_content_chars - head
                    message["content"] = (
                        content[:head]
                        + "\n...[tool page truncated]...\n"
                        + content[-tail:]
                    )
        item = {
            "task_id": row["task_id"],
            "trajectory": trajectory,
        }
        size = len(json.dumps(item, ensure_ascii=False))
        if current and (len(current) >= max_rows or current_chars + size > max_chars):
            batches.append(current)
            current, current_chars = [], 0
        current.append(item)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def validate_labels(result: dict, expected_ids: set[int]) -> list[dict]:
    labels = result.get("labels")
    if not isinstance(labels, list):
        raise ValueError("response.labels must be a list")
    if {x.get("task_id") for x in labels if isinstance(x, dict)} != expected_ids:
        raise ValueError("response task_ids do not match request")
    required = {
        "task_id",
        "difficulty",
        "trajectory_complexity",
        "domain",
        "price_band",
        "needs_query_rewrite",
        "needs_candidate_comparison",
        "variant_complexity",
        "budget_risk",
        "loop_risk",
        "confidence",
        "evidence",
    }
    for label in labels:
        if required.difference(label):
            raise ValueError(f"label {label.get('task_id')} is missing required fields")
        if label["difficulty"] not in LEVELS or label["trajectory_complexity"] not in LEVELS:
            raise ValueError(f"label {label['task_id']} has an invalid difficulty")
        if label["price_band"] not in {"low", "medium", "high", "unknown"}:
            raise ValueError(f"label {label['task_id']} has an invalid price_band")
        for key in ("variant_complexity", "budget_risk", "loop_risk"):
            if label[key] not in {"low", "medium", "high"}:
                raise ValueError(f"label {label['task_id']} has an invalid {key}")
        if not 0 <= float(label["confidence"]) <= 1:
            raise ValueError(f"label {label['task_id']} has invalid confidence")
    return labels


def label_batch(batch: list[dict], model: str) -> tuple[list[dict], dict]:
    client = client_from_environment(
        model=model,
        max_tokens=2200,
        timeout=90,
        retries=2,
        response_format_json=True,
        thinking=False,
    )
    response = client.complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"tasks": batch}, ensure_ascii=False)},
        ]
    )
    expected_ids = {int(item["task_id"]) for item in batch}
    return validate_labels(response["result"], expected_ids), response["metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-batch-chars", type=int, default=60000)
    parser.add_argument("--max-tool-content-chars", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = (
        {row["task_id"] for row in read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    pending = [row for row in rows if row["task_id"] not in existing]
    batches = make_batches(
        pending, args.batch_size, args.max_batch_chars, args.max_tool_content_chars
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    failures = []
    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(label_batch, batch, args.model): batch for batch in batches}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    labels, metadata = future.result()
                except Exception as exc:  # Keep successful batches resumable.
                    task_ids = [item["task_id"] for item in futures[future]]
                    failures.append(
                        {
                            "task_ids": task_ids,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        json.dumps(
                            {"failed_batch": task_ids, "error": str(exc)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                for label in sorted(labels, key=lambda item: item["task_id"]):
                    text = json.dumps(
                        label, ensure_ascii=False, separators=(",", ":")
                    )
                    handle.write(text + "\n")
                handle.flush()
                for key in usage:
                    usage[key] += int((metadata.get("usage") or {}).get(key) or 0)
                print(
                    json.dumps(
                        {
                            "completed_batches": index,
                            "total_batches": len(batches),
                            "labels": len(labels),
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            {
                "existing": len(existing),
                "requested": len(pending),
                "failures": failures,
                "usage": usage,
            },
            ensure_ascii=False,
        )
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
