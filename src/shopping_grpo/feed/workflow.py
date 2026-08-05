"""One-command CPU workflow: data → frozen evaluation → interactive demo."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from shopping_grpo.feed.datasets import generate_feed_artifacts
from shopping_grpo.feed.demo import write_demo
from shopping_grpo.feed.manifest import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    verify_manifest,
    write_json_atomic,
)
from shopping_grpo.feed.report import evaluate_log_file


def run_cpu_mvp(
    catalog: Any,
    output_dir: str | Path,
    *,
    episodes: int = 30,
    feed_length: int = 24,
    seed: int = 42,
    force: bool = False,
    calibration: Any = None,
) -> dict[str, Any]:
    """Materialize the complete non-model workflow without starting training."""
    root = Path(output_dir)
    dataset_manifest = generate_feed_artifacts(
        catalog,
        root,
        episodes=episodes,
        feed_length=feed_length,
        seed=seed,
        force=force,
        calibration=calibration,
    )
    verify_manifest(root)
    logs = root / "mixed_policy_logs" / "test.jsonl"
    evaluation_dir = root / "evaluation"
    report = evaluate_log_file(
        logs,
        evaluation_dir,
        split="test",
        dataset_dir=root,
        require_paired=True,
    )
    dashboard = write_demo(
        logs,
        evaluation_dir / "dashboard.html",
        evaluation_summary_path=evaluation_dir / "report.json",
        title="Feed Agent Lab · Frozen Test",
    )
    declared = {
        "dataset_manifest": root / "manifest.json",
        "evaluation_manifest": evaluation_dir / "evaluation_manifest.json",
        "demo_manifest": dashboard.with_suffix(".manifest.json"),
        "dashboard": dashboard,
    }
    workflow_manifest: dict[str, Any] = {
        "schema_version": "feed-cpu-workflow-v1",
        "training_started": False,
        "llm_judge": False,
        "config": {
            "episodes": int(episodes),
            "feed_length": int(feed_length),
            "seed": int(seed),
            "calibrated": calibration is not None,
        },
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in declared.items()
        },
        "dataset_manifest_content_sha256": dataset_manifest[
            "manifest_content_sha256"
        ],
        "evaluation_report_content_sha256": report["report_content_sha256"],
    }
    workflow_manifest["manifest_content_sha256"] = sha256_bytes(
        canonical_json(workflow_manifest).encode("utf-8")
    )
    write_json_atomic(root / "workflow_manifest.json", workflow_manifest)
    return workflow_manifest


def _verify_content_hash(payload: dict[str, Any], field: str, label: str) -> str:
    expected = payload.pop(field, None)
    actual = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if expected != actual:
        raise ValueError(f"{label} content hash mismatch")
    payload[field] = expected
    return str(expected)


def verify_cpu_mvp(output_dir: str | Path) -> dict[str, Any]:
    """Reopen and hash-check the complete dataset/evaluation/demo workflow."""
    root = Path(output_dir)
    dataset = verify_manifest(root)
    workflow_path = root / "workflow_manifest.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    _verify_content_hash(workflow, "manifest_content_sha256", "workflow manifest")
    for name, metadata in workflow.get("artifacts", {}).items():
        path = root / metadata["path"]
        if not path.is_file():
            raise ValueError(f"workflow artifact is missing: {name}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"workflow artifact byte size mismatch: {name}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"workflow artifact hash mismatch: {name}")

    evaluation_path = root / "evaluation" / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    _verify_content_hash(
        evaluation, "manifest_content_sha256", "evaluation manifest"
    )
    for name, metadata in evaluation.get("files", {}).items():
        path = evaluation_path.parent / name
        if path.stat().st_size != int(metadata["bytes"]) or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"evaluation artifact mismatch: {name}")

    demo_path = root / "evaluation" / "dashboard.manifest.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    _verify_content_hash(demo, "manifest_content_sha256", "demo manifest")
    dashboard = demo_path.parent / demo["output"]["path"]
    if (
        dashboard.stat().st_size != int(demo["output"]["bytes"])
        or sha256_file(dashboard) != demo["output"]["sha256"]
    ):
        raise ValueError("demo output mismatch")
    return {
        "ok": True,
        "schema_version": workflow["schema_version"],
        "dataset_manifest_content_sha256": dataset["manifest_content_sha256"],
        "workflow_manifest_content_sha256": workflow["manifest_content_sha256"],
    }


__all__ = ["run_cpu_mvp", "verify_cpu_mvp"]
