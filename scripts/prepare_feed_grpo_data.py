#!/usr/bin/env python3
"""Convert self-contained Feed online-RL JSONL tasks to veRL Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from shopping_grpo.feed.manifest import (
    canonical_json,
    sha256_file,
    verify_manifest,
    write_json_atomic,
)
from shopping_grpo.feed.schema import iter_jsonl


SPLITS = ("train", "validation")
ALL_SPLITS = (*SPLITS, "test")
JSON_ENCODED_FIELDS = ("episode_seed", "catalog_products", "calibration")


def normalize_task_for_parquet(row: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-encode dynamic nested truth while preserving veRL prompt structures."""
    normalized = json.loads(canonical_json(row))
    for field in JSON_ENCODED_FIELDS:
        if field in normalized and not isinstance(normalized[field], str):
            normalized[field] = canonical_json(normalized[field])
    extra = normalized.get("extra_info")
    if isinstance(extra, dict) and "episode_seed" in extra and not isinstance(
        extra["episode_seed"], str
    ):
        extra["episode_seed"] = canonical_json(extra["episode_seed"])
    return normalized


def inspect_input(path: str | Path) -> dict[str, Any]:
    rows = [normalize_task_for_parquet(row) for row in iter_jsonl(path)]
    if not rows:
        raise ValueError(f"online RL task file is empty: {path}")
    required = {"data_source", "task_id", "prompt", "episode_seed", "catalog_products"}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {index} is missing: {', '.join(missing)}")
        if not isinstance(row["prompt"], list):
            raise ValueError(f"row {index} prompt must be a message list")
    return {
        "rows": rows,
        "row_count": len(rows),
        "source_sha256": sha256_file(path),
        "identities": sorted(_task_identity(row) for row in rows),
    }


def _task_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    raw_seed = row.get("episode_seed")
    seed = json.loads(raw_seed) if isinstance(raw_seed, str) else raw_seed
    persona = seed.get("persona") if isinstance(seed, Mapping) else None
    episode_id = str(row.get("episode_id") or row.get("task_id") or "")
    persona_id = str(
        persona.get("persona_id") if isinstance(persona, Mapping) else ""
    )
    if not episode_id or not persona_id:
        raise ValueError("online RL row lacks episode/persona identity")
    return episode_id, persona_id


def write_parquet(rows: list[dict[str, Any]], destination: str | Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # training extras provide pyarrow through veRL/Ray
        raise RuntimeError(
            "Parquet conversion requires pyarrow from the GRPO training environment"
        ) from exc
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    inspect_only: bool = False,
) -> dict[str, Any]:
    source_root, target_root = Path(input_dir), Path(output_dir)
    split_reports: dict[str, Any] = {}
    for split in ALL_SPLITS:
        source = source_root / f"{split}.jsonl"
        inspected = inspect_input(source)
        report = {
            "source": source.name,
            "source_sha256": inspected["source_sha256"],
            "rows": inspected["row_count"],
            "identities": [list(item) for item in inspected["identities"]],
        }
        if split in SPLITS and not inspect_only:
            destination = write_parquet(
                inspected["rows"], target_root / f"{split}.parquet"
            )
            report.update(
                {
                    "output": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        split_reports[split] = report
    identity_sets = {
        split: {tuple(item) for item in split_reports[split]["identities"]}
        for split in ALL_SPLITS
    }
    for left_index, left in enumerate(ALL_SPLITS):
        for right in ALL_SPLITS[left_index + 1 :]:
            left_episodes = {item[0] for item in identity_sets[left]}
            right_episodes = {item[0] for item in identity_sets[right]}
            left_personas = {item[1] for item in identity_sets[left]}
            right_personas = {item[1] for item in identity_sets[right]}
            if left_episodes & right_episodes or left_personas & right_personas:
                raise ValueError(f"Feed split identities overlap: {left}/{right}")
    dataset_root = source_root.parent
    dataset_manifest = verify_manifest(dataset_root)
    for split in ALL_SPLITS:
        relative = f"online_rl_tasks/{split}.jsonl"
        declared = dataset_manifest.get("files", {}).get(relative)
        if (
            not isinstance(declared, Mapping)
            or declared.get("sha256") != split_reports[split]["source_sha256"]
        ):
            raise ValueError(f"dataset manifest does not bind {relative}")
    manifest = {
        "schema_version": "feed-grpo-parquet-manifest-v1",
        "inspect_only": bool(inspect_only),
        "json_encoded_fields": list(JSON_ENCODED_FIELDS),
        "dataset_manifest_content_sha256": dataset_manifest[
            "manifest_content_sha256"
        ],
        "splits": split_reports,
    }
    if not inspect_only:
        manifest["manifest_content_sha256"] = hashlib.sha256(
            canonical_json(manifest).encode("utf-8")
        ).hexdigest()
        write_json_atomic(target_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="online_rl_tasks directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="validate JSONL without importing pyarrow or writing Parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            convert_directory(
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
