#!/usr/bin/env python3
"""Seal model Feed rollouts to a frozen dataset and checkpoint identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from shopping_grpo.feed.manifest import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    verify_manifest,
    write_json_atomic,
)
from shopping_grpo.feed.schema import iter_jsonl


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    target = path.expanduser().resolve()
    if target.is_file():
        return {
            "name": target.name,
            "kind": "file",
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    if not target.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {target}")
    files = []
    for file_path in sorted(item for item in target.rglob("*") if item.is_file()):
        relative = file_path.relative_to(target).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    if not files:
        raise ValueError("checkpoint directory is empty")
    return {
        "name": target.name,
        "kind": "directory",
        "files": len(files),
        "sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def seal_run(
    logs_path: str | Path,
    dataset_dir: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    policy_id: str,
    split: str = "test",
) -> dict[str, Any]:
    if split != "test":
        raise ValueError("frozen model rollout sealing only accepts split=test")
    policy_id = str(policy_id).strip()
    if not policy_id:
        raise ValueError("policy_id must not be empty")
    logs = Path(logs_path)
    root = Path(dataset_dir).resolve()
    dataset_manifest = verify_manifest(root)
    seeds = root / "seeds" / f"{split}.jsonl"
    declared_seeds = dataset_manifest.get("files", {}).get(f"seeds/{split}.jsonl")
    if not isinstance(declared_seeds, Mapping) or declared_seeds.get("sha256") != sha256_file(seeds):
        raise ValueError("dataset manifest does not bind frozen seeds")
    expected = {}
    for row in iter_jsonl(seeds):
        persona = row.get("persona") or {}
        expected[str(row.get("episode_id") or "")] = str(
            persona.get("persona_id") if isinstance(persona, Mapping) else ""
        )
    observed = {}
    for index, row in enumerate(iter_jsonl(logs)):
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        row_split = str(row.get("split") or metadata.get("split") or "")
        episode_id = str(row.get("episode_id") or row.get("task_id") or "")
        persona_id = str(row.get("persona_id") or "")
        policy = str(row.get("policy") or row.get("policy_name") or row.get("behavior_policy") or "")
        if row_split != split or policy != policy_id:
            raise ValueError(f"rollout row {index} has wrong split or policy")
        if not episode_id or episode_id in observed:
            raise ValueError(f"rollout row {index} has missing/duplicate episode")
        observed[episode_id] = persona_id
    if observed != expected:
        raise ValueError("model rollouts do not exactly match frozen episode/persona IDs")
    run = {
        "schema_version": "feed-frozen-rollout-run-v1",
        "split": split,
        "policy_id": policy_id,
        "logs": logs.name,
        "logs_sha256": sha256_file(logs),
        "dataset_manifest_content_sha256": dataset_manifest[
            "manifest_content_sha256"
        ],
        "seeds_sha256": sha256_file(seeds),
        "checkpoint": _checkpoint_identity(Path(checkpoint)),
        "llm_judge": False,
    }
    run["manifest_content_sha256"] = hashlib.sha256(
        canonical_json(run).encode("utf-8")
    ).hexdigest()
    write_json_atomic(output, run)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            seal_run(
                args.logs,
                args.dataset_dir,
                args.checkpoint,
                args.output,
                policy_id=args.policy_id,
                split=args.split,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
