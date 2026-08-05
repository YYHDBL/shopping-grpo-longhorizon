#!/usr/bin/env python3
"""Validate Feed CRN preference pairs and emit conversational DPO JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from shopping_grpo.feed.manifest import canonical_json, sha256_file, write_json_atomic
from shopping_grpo.feed.schema import iter_jsonl, write_jsonl


TRAIN_SPLITS = ("train", "validation")
ALL_SPLITS = (*TRAIN_SPLITS, "test")


def _completion(message: Any, *, label: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise ValueError(f"{label} must be one assistant message")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError(f"{label} must contain exactly one tool call")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != "commit_recommendation":
        raise ValueError(f"{label} must call commit_recommendation")
    call_id = str(call.get("id") or "")
    if not call_id or "chosen" in call_id.casefold() or "rejected" in call_id.casefold():
        raise ValueError(f"{label} call ID leaks the preference label")
    try:
        arguments = json.loads(function["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} has invalid JSON arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"{label} arguments must be an object")
    return dict(message), call_id, arguments


def normalize_pair(row: Mapping[str, Any], *, split: str, index: int) -> dict[str, Any]:
    if row.get("split") != split:
        raise ValueError(f"{split} row {index} declares a different split")
    prompt = row.get("prompt")
    tools = row.get("tools")
    if not isinstance(prompt, list) or not prompt or not isinstance(tools, list):
        raise ValueError(f"{split} row {index} requires prompt and tools")
    chosen, chosen_id, chosen_args = _completion(row.get("chosen"), label="chosen")
    rejected, rejected_id, rejected_args = _completion(row.get("rejected"), label="rejected")
    if chosen_id != rejected_id:
        raise ValueError(f"{split} row {index} completion call IDs must be neutral and equal")
    if chosen_args != row.get("chosen_action") or rejected_args != row.get("rejected_action"):
        raise ValueError(f"{split} row {index} action labels do not match completions")
    margin = row.get("return_margin")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
        raise ValueError(f"{split} row {index} has invalid return_margin")
    if not row.get("common_random_numbers") or row.get("counterfactual_method") != "common_random_numbers":
        raise ValueError(f"{split} row {index} is not a CRN preference pair")
    episode_id = str(row.get("episode_id") or "")
    persona_id = str(row.get("persona_id") or "")
    if not episode_id or not persona_id:
        raise ValueError(f"{split} row {index} lacks episode/persona identity")
    return {
        "pair_id": str(row.get("pair_id") or ""),
        "episode_id": episode_id,
        "persona_id": persona_id,
        "split": split,
        "prompt": prompt,
        "chosen": [chosen],
        "rejected": [rejected],
        "tools": tools,
        "return_margin": float(margin),
        "common_random_seed": row.get("common_random_seed"),
        "counterfactual_method": "common_random_numbers",
    }


def prepare_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    inspect_only: bool = False,
) -> dict[str, Any]:
    source_root = Path(input_dir)
    target_root = Path(output_dir)
    identities: dict[str, set[tuple[str, str]]] = {}
    reports: dict[str, Any] = {}
    for split in ALL_SPLITS:
        source = source_root / f"{split}.jsonl"
        rows = [
            normalize_pair(row, split=split, index=index)
            for index, row in enumerate(iter_jsonl(source))
        ]
        if not rows:
            raise ValueError(f"{split} preference input is empty")
        identities[split] = {
            (row["episode_id"], row["persona_id"])
            for row in rows
        }
        report: dict[str, Any] = {
            "source": source.name,
            "source_sha256": sha256_file(source),
            "rows": len(rows),
        }
        if split in TRAIN_SPLITS and not inspect_only:
            destination = target_root / f"{split}.jsonl"
            write_jsonl(destination, rows)
            report.update(
                {
                    "output": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        reports[split] = report
    for left_index, left in enumerate(ALL_SPLITS):
        for right in ALL_SPLITS[left_index + 1 :]:
            if identities[left] & identities[right]:
                raise ValueError(f"preference identities overlap: {left}/{right}")
    manifest = {
        "schema_version": "feed-dpo-conversational-v1",
        "inspect_only": bool(inspect_only),
        "format": "prompt + chosen/rejected assistant-message arrays + tools",
        "splits": reports,
    }
    if not inspect_only:
        manifest["manifest_content_sha256"] = hashlib.sha256(
            canonical_json(manifest).encode("utf-8")
        ).hexdigest()
        write_json_atomic(target_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            prepare_directory(
                args.input_dir,
                args.output_dir,
                inspect_only=args.inspect_only,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
