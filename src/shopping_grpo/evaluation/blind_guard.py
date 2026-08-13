"""Content- and task-ID protection for the frozen blind final test."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path

from shopping_grpo.evaluation.artifacts import ArtifactError

BLIND_GUARD_SCHEMA = "shopping-blind-asset-guard-v1"
BLIND_TASK_IDS_SCHEMA = "shopping-blind-task-ids-v1"
_RESOURCE_PACKAGE = "shopping_grpo.resources"
_GUARD_RESOURCE = "blind_guard.json"
_EXPECTED_METADATA = {
    "asset": "shop_benchmark_reward_v3_final_200_clean",
    "contract": "environment-v2.1/reward-v3/curated-final200-clean-v1",
    "environment_version": "shopsimulator-environment-v2.1",
    "reward_version": "shopsimulator-reward-v3",
    "evaluated": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_resource_object(name: str) -> dict:
    try:
        resource = files(_RESOURCE_PACKAGE).joinpath(name)
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (ModuleNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read packaged blind resource: {name}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"packaged blind resource must be an object: {name}")
    return value


def _row_task_id(row: Mapping) -> int | None:
    value = row.get("task_id")
    if value is None:
        extra = row.get("extra_info")
        if isinstance(extra, Mapping):
            value = extra.get("task_id")
    if value is None:
        normalized = row.get("normalized_trajectory")
        if isinstance(normalized, Mapping):
            value = normalized.get("task_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid task_id in {row!r}") from exc


def _jsonl_task_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    opener = gzip.open if path.name.endswith(".gz") else open
    task_ids = set()
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArtifactError(
                        f"{path}:{line_number}: JSONL row must be an object"
                    )
                task_id = _row_task_id(value)
                if task_id is not None:
                    task_ids.add(task_id)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSONL during blind guard") from exc
    return task_ids


def validate_canonical_blind_asset() -> tuple[dict, set[int]]:
    """Validate the wheel-packaged guard contract and frozen task-ID set."""

    guard = _load_resource_object(_GUARD_RESOURCE)
    if guard.get("schema_version") != BLIND_GUARD_SCHEMA:
        raise ArtifactError("unsupported blind guard schema")
    if guard.get("manifest_version") != 1:
        raise ArtifactError("unsupported blind guard manifest version")
    if guard.get("split_role") != "blind_final_test":
        raise ArtifactError("blind guard split_role must be blind_final_test")
    required = guard.get("required_metadata")
    if not isinstance(required, Mapping):
        raise ArtifactError("blind guard required_metadata must be an object")
    mismatches = {
        key: {"required": expected, "actual": required.get(key)}
        for key, expected in _EXPECTED_METADATA.items()
        if required.get(key) != expected
    }
    if mismatches:
        raise ArtifactError(
            "packaged blind metadata contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    for field in ("task_sha256", "metadata_sha256"):
        digest = guard.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ArtifactError(f"invalid packaged blind {field}")

    ids_resource = guard.get("task_ids_resource")
    if not isinstance(ids_resource, str) or not ids_resource:
        raise ArtifactError("blind guard task_ids_resource is missing")
    ids_document = _load_resource_object(ids_resource)
    if ids_document.get("schema_version") != BLIND_TASK_IDS_SCHEMA:
        raise ArtifactError("unsupported blind task-ID schema")
    raw_task_ids = ids_document.get("task_ids")
    if not isinstance(raw_task_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in raw_task_ids
    ):
        raise ArtifactError("packaged blind task_ids must be integers")
    task_ids = set(raw_task_ids)
    if len(task_ids) != len(raw_task_ids):
        raise ArtifactError("packaged blind task_ids contain duplicates")
    if len(task_ids) != int(guard.get("task_count", -1)):
        raise ArtifactError("packaged blind task count mismatch")
    return guard, task_ids


def guard_blind_final(
    paths: Iterable[Path],
    *,
    allowed: bool,
) -> None:
    """Reject any artifact containing final-test task IDs, independent of name."""

    guard, final_task_ids = validate_canonical_blind_asset()
    if allowed:
        return
    blocked = {}
    canonical_sha = str(guard["task_sha256"])
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        same_content = _sha256_file(path) == canonical_sha
        overlap = sorted(_jsonl_task_ids(path) & final_task_ids)
        if same_content or overlap:
            blocked[str(path)] = {
                "same_content": same_content,
                "overlap_count": len(overlap),
                "sample_task_ids": overlap[:10],
            }
    if blocked:
        raise ArtifactError(
            "refusing to consume frozen blind-final tasks without "
            "--allow-blind-final: "
            + json.dumps(blocked, ensure_ascii=False, sort_keys=True)
        )
