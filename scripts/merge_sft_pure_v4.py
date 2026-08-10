#!/usr/bin/env python3
"""Merge the two DeepSeek-v4 SFT sources without task leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def read_git_jsonl(repo: Path, ref: str, relative_path: str) -> list[dict]:
    text = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ref}:{relative_path}"],
        text=True,
    )
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def content_hash(row: dict) -> str:
    payload = {"messages": row.get("messages"), "tools": row.get("tools")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tool_calls(row: dict) -> list[str]:
    calls = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            calls.append(str(function.get("name") or ""))
    return calls


def quality_features(row: dict) -> dict:
    calls = _tool_calls(row)
    guard_markers = ("guard rejected", "action guard", "动作守卫拒绝", "guard_violation")
    guard_rejections = sum(
        marker in str(message.get("content") or "").casefold()
        for message in row.get("messages") or []
        if message.get("role") == "tool"
        for marker in guard_markers
    )
    return {
        "guard_rejections": int(guard_rejections),
        "gold_purchase": "buy_now" in calls,
        "steps": len(calls),
        "searches": calls.count("search_products"),
        "opens": calls.count("open_product"),
        "unique_candidates": len(
            {
                str(message.get("content") or "").split("asin:", 1)[1].splitlines()[0].strip()
                for message in row.get("messages") or []
                if message.get("role") == "tool" and "asin:" in str(message.get("content") or "")
            }
        ),
    }


def _tie_rank(seed: int, task_id: int, row_hash: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}:{row_hash}".encode("utf-8")).hexdigest()


def _candidate_key(row: dict, *, seed: int) -> tuple:
    features = quality_features(row)
    row_hash = content_hash(row)
    return (
        features["guard_rejections"] == 0,
        features["gold_purchase"],
        -features["steps"],
        _tie_rank(seed, int(row["task_id"]), row_hash),
    )


def _validate_row(row: dict, source: str) -> None:
    required = {"task_id", "trajectory_id", "messages", "tools"}
    missing = required.difference(row)
    if missing:
        raise ValueError(f"{source}: missing fields {sorted(missing)}")
    if not isinstance(row["task_id"], int) or isinstance(row["task_id"], bool):
        raise ValueError(f"{source}: task_id must be an integer")
    if not isinstance(row["messages"], list) or not isinstance(row["tools"], list):
        raise ValueError(f"{source}: messages/tools must be lists")


def merge_sources(
    *,
    current_train: Path,
    current_validation: Path,
    merged_train: Path,
    merged_validation: Path,
    legacy_rows: Iterable[dict],
    seed: int,
) -> tuple[list[dict], dict]:
    legacy_hashes = {content_hash(row) for row in legacy_rows}
    candidates = []
    for path, source in (
        (current_train, "current_v4_train"),
        (current_validation, "current_v4_validation"),
        (merged_train, "merged_v4_train"),
        (merged_validation, "merged_v4_validation"),
    ):
        for row in read_jsonl(path):
            _validate_row(row, f"{source}:{row.get('task_id')}")
            if source.startswith("merged") and content_hash(row) in legacy_hashes:
                continue
            candidates.append({"row": row, "source": source, "content_hash": content_hash(row)})

    by_task: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_task.setdefault(int(candidate["row"]["task_id"]), []).append(candidate)

    selected = []
    duplicates = []
    for task_id in sorted(by_task):
        group = by_task[task_id]
        winner = max(group, key=lambda item: _candidate_key(item["row"], seed=seed))
        selected.append(winner)
        for candidate in group:
            if candidate is winner:
                continue
            duplicates.append(
                {
                    "task_id": task_id,
                    "selected_source": winner["source"],
                    "selected_content_hash": winner["content_hash"],
                    "discarded_source": candidate["source"],
                    "discarded_content_hash": candidate["content_hash"],
                    "selected_quality": quality_features(winner["row"]),
                    "discarded_quality": quality_features(candidate["row"]),
                }
            )

    selected.sort(key=lambda item: (int(item["row"]["task_id"]), item["content_hash"]))
    report = {
        "schema_version": "shopping-pure-v4-sft-merge-v1",
        "seed": int(seed),
        "legacy_rows_removed_from_merged": len(legacy_hashes),
        "candidate_rows_after_legacy_removal": len(candidates),
        "selected_rows": len(selected),
        "duplicate_task_groups": sum(len(group) > 1 for group in by_task.values()),
        "discarded_duplicate_rows": len(duplicates),
        "source_rows": {
            source: sum(item["source"] == source for item in candidates)
            for source in (
                "current_v4_train",
                "current_v4_validation",
                "merged_v4_train",
                "merged_v4_validation",
            )
        },
        "duplicates": duplicates,
        "selected": [
            {
                "task_id": int(item["row"]["task_id"]),
                "trajectory_id": item["row"]["trajectory_id"],
                "source": item["source"],
                "content_hash": item["content_hash"],
                "quality": quality_features(item["row"]),
            }
            for item in selected
        ],
    }
    return [item["row"] for item in selected], report


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--current-train", type=Path, required=True)
    parser.add_argument("--current-validation", type=Path, required=True)
    parser.add_argument("--merged-train", type=Path, required=True)
    parser.add_argument("--merged-validation", type=Path, required=True)
    parser.add_argument("--legacy-ref", default="HEAD^")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_rows = read_git_jsonl(args.repo, args.legacy_ref, "data/sft/train.jsonl")
    legacy_rows += read_git_jsonl(args.repo, args.legacy_ref, "data/sft/validation.jsonl")
    rows, report = merge_sources(
        current_train=args.current_train,
        current_validation=args.current_validation,
        merged_train=args.merged_train,
        merged_validation=args.merged_validation,
        legacy_rows=legacy_rows,
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "all.jsonl", rows)
    (args.output_dir / "duplicate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"duplicates", "selected"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
