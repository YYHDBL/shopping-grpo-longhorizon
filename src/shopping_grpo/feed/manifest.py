"""Versioned manifests and leakage guards for generated Feed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


FEED_PROFILE_VERSIONS = {
    "environment": "feed-environment-v1",
    "observation": "feed-observation-v1",
    "tools": "feed-tools-v1",
    "reward": "feed-reward-v1",
    "artifact_schema": "feed-artifacts-v1",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def audit_split_isolation(
    splits: Mapping[str, Iterable[Any]],
) -> dict[str, dict[str, int]]:
    """Require disjoint episode and persona identifiers across every split."""
    episode_owners: dict[str, str] = {}
    persona_owners: dict[str, str] = {}
    summary: dict[str, dict[str, int]] = {}
    for split_name in sorted(splits):
        rows = list(splits[split_name])
        episode_ids: set[str] = set()
        persona_ids: set[str] = set()
        for raw in rows:
            row = raw.to_dict() if hasattr(raw, "to_dict") else raw
            if not isinstance(row, Mapping):
                raise TypeError(f"{split_name} rows must be objects")
            episode_id = str(row.get("episode_id") or "")
            persona = row.get("persona") or {}
            persona_id = str(
                (persona.get("persona_id") or persona.get("user_id") or "")
                if isinstance(persona, Mapping)
                else getattr(persona, "persona_id", "")
            )
            if not episode_id or not persona_id:
                raise ValueError(f"{split_name} rows require episode_id and persona_id")
            previous = episode_owners.setdefault(episode_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"episode_id {episode_id!r} overlaps {previous!r} and {split_name!r}"
                )
            previous = persona_owners.setdefault(persona_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"persona_id {persona_id!r} overlaps {previous!r} and {split_name!r}"
                )
            episode_ids.add(episode_id)
            persona_ids.add(persona_id)
        summary[split_name] = {
            "rows": len(rows),
            "episodes": len(episode_ids),
            "personas": len(persona_ids),
        }
    return summary


def build_manifest(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    source_catalog: Mapping[str, Any] | None = None,
    include_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Hash the selected generated files and return an audit manifest.

    ``include_paths`` scopes a component manifest to its own files.  This keeps
    repeatable workflows valid when reports or dashboards share ``output_dir``.
    """
    root = Path(output_dir)
    candidates: set[Path]
    if include_paths is None:
        candidates = {item for item in root.rglob("*") if item.is_file()}
    else:
        candidates = set()
        for raw_path in include_paths:
            relative_path = Path(raw_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"include path must stay below output_dir: {raw_path}")
            selected = root / relative_path
            if selected.is_file():
                candidates.add(selected)
            elif selected.is_dir():
                candidates.update(item for item in selected.rglob("*") if item.is_file())
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(candidates):
        if path.name == "manifest.json" or path.name.startswith("."):
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "feed-artifact-manifest-v1",
        "profile_versions": dict(FEED_PROFILE_VERSIONS),
        "config": json.loads(canonical_json(config)),
        "split_isolation": json.loads(canonical_json(split_summary)),
        "source_catalog": json.loads(canonical_json(source_catalog or {})),
        "files": files,
    }
    manifest["manifest_content_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    return manifest


def verify_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Reopen a manifest and verify all declared artifact hashes."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_content_hash = manifest.pop("manifest_content_sha256", None)
    if expected_content_hash != sha256_bytes(canonical_json(manifest).encode("utf-8")):
        raise ValueError("manifest content hash mismatch")
    for relative, metadata in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"manifest file is missing: {relative}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"manifest byte size mismatch: {relative}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"manifest SHA-256 mismatch: {relative}")
    manifest["manifest_content_sha256"] = expected_content_hash
    return manifest
