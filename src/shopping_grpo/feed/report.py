"""Frozen, paired and code-only evaluation reports for Feed trajectories."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from shopping_grpo.feed.evaluation import evaluate_episodes
from shopping_grpo.feed.manifest import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    verify_manifest,
    write_json_atomic,
)
from shopping_grpo.feed.schema import iter_jsonl


REPORT_SCHEMA_VERSION = "feed-frozen-evaluation-v1"
REPORT_METRICS = (
    "long_term_return",
    "qualified_purchase_rate",
    "correct_no_recommend_rate",
    "interventions_per_100",
    "return_rate",
    "irrelevant_recommendation_rate",
    "repeat_exposure_rate",
    "grounded_recommendation_rate",
    "unsupported_claim_rate",
    "mean_dwell_seconds",
    "skip_rate",
    "terminal_satisfaction",
    "complementary_bundle_precision",
    "net_revenue",
    "terminal_fatigue",
)


def _split(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return str(row.get("split") or metadata.get("split") or "")


def _episode_id(row: Mapping[str, Any]) -> str:
    nested = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    return str(row.get("episode_id") or row.get("task_id") or nested.get("episode_id") or "")


def _policy(row: Mapping[str, Any]) -> str:
    return str(
        row.get("policy")
        or row.get("policy_name")
        or row.get("behavior_policy")
        or ""
    )


def build_frozen_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = "test",
    require_paired: bool = True,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen split and require every policy to see the same episodes."""
    selected = [dict(row) for row in rows if _split(row) == split]
    if not selected:
        raise ValueError(f"no trajectory rows found for frozen split {split!r}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in selected:
        policy = _policy(row)
        episode_id = _episode_id(row)
        if not policy or not episode_id:
            raise ValueError("each evaluation row requires policy and episode_id")
        key = (policy, episode_id)
        if key in seen:
            raise ValueError(f"duplicate policy/episode trajectory: {policy}/{episode_id}")
        seen.add(key)
        grouped[policy].append(row)
    episode_sets = {policy: {_episode_id(row) for row in runs} for policy, runs in grouped.items()}
    if require_paired:
        reference_policy = sorted(episode_sets)[0]
        reference = episode_sets[reference_policy]
        for policy, episodes in episode_sets.items():
            if episodes != reference:
                raise ValueError(
                    f"unpaired frozen evaluation: {policy!r} differs from {reference_policy!r}"
                )

    policy_metrics = {}
    for policy in sorted(grouped):
        runs = sorted(grouped[policy], key=_episode_id)
        policy_metrics[policy] = evaluate_episodes(runs)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "split": split,
        "paired": bool(require_paired),
        "llm_judge": False,
        "policy_count": len(grouped),
        "episode_count_per_policy": {
            policy: len(episodes) for policy, episodes in sorted(episode_sets.items())
        },
        "episode_ids": sorted(set.intersection(*episode_sets.values())),
        "source": dict(source or {}),
        "policy_metrics": policy_metrics,
    }
    report["report_content_sha256"] = sha256_bytes(canonical_json(report).encode("utf-8"))
    return report


def render_report_markdown(report: Mapping[str, Any]) -> str:
    policies = report.get("policy_metrics")
    if not isinstance(policies, Mapping):
        raise ValueError("report has no policy_metrics")
    available = [
        metric
        for metric in REPORT_METRICS
        if any(isinstance(row, Mapping) and metric in row for row in policies.values())
    ]
    lines = [
        "# Feed frozen evaluation",
        "",
        f"- Schema: `{report.get('schema_version')}`",
        f"- Split: `{report.get('split')}`",
        f"- Paired policies: `{str(bool(report.get('paired'))).lower()}`",
        "- Scoring: deterministic simulator metrics; no LLM judge",
        "",
        "| Policy | " + " | ".join(available) + " |",
        "|---|" + "---:|" * len(available),
    ]
    for policy in sorted(policies):
        metrics = policies[policy]
        values = []
        for metric in available:
            value = metrics.get(metric, 0.0) if isinstance(metrics, Mapping) else 0.0
            values.append(f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value))
        lines.append(f"| {policy} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "> Reward is not collapsed with experience or safety metrics; inspect every column.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def evaluate_log_file(
    logs_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    dataset_dir: str | Path | None = None,
    run_manifest: str | Path | None = None,
    require_paired: bool = True,
) -> dict[str, Any]:
    logs = Path(logs_path)
    rows = list(iter_jsonl(logs))
    source: dict[str, Any] = {
        "logs": logs.name,
        "logs_sha256": sha256_file(logs),
    }
    expected_personas: dict[str, str] | None = None
    if dataset_dir is not None:
        dataset_root = Path(dataset_dir).resolve()
        dataset_manifest = verify_manifest(dataset_root)
        expected_logs = (dataset_root / "mixed_policy_logs" / f"{split}.jsonl").resolve()
        logs_relative = expected_logs.relative_to(dataset_root).as_posix()
        if logs.resolve() == expected_logs:
            declared_logs = dataset_manifest.get("files", {}).get(logs_relative)
            if not isinstance(declared_logs, Mapping):
                raise ValueError(
                    "frozen evaluation logs are not declared by the dataset manifest"
                )
            if declared_logs.get("sha256") != source["logs_sha256"]:
                raise ValueError("frozen evaluation logs do not match the dataset manifest")
            source["run_type"] = "manifest_bound_baseline"
        else:
            run = _verify_frozen_run_manifest(
                run_manifest,
                logs=logs,
                split=split,
                dataset_manifest=dataset_manifest,
            )
            source["run_type"] = "sealed_model_rollout"
            source["run_manifest"] = Path(run_manifest).name
            source["run_manifest_content_sha256"] = run[
                "manifest_content_sha256"
            ]
            source["policy_id"] = run["policy_id"]
            source["checkpoint"] = run["checkpoint"]

        seeds = dataset_root / "seeds" / f"{split}.jsonl"
        seeds_relative = seeds.relative_to(dataset_root).as_posix()
        declared_seeds = dataset_manifest.get("files", {}).get(seeds_relative)
        if not isinstance(declared_seeds, Mapping):
            raise ValueError(
                "frozen evaluation seeds are not declared by the dataset manifest"
            )
        expected_personas = {}
        for seed_row in iter_jsonl(seeds):
            episode_id = str(seed_row.get("episode_id") or "")
            persona = seed_row.get("persona") or {}
            persona_id = str(
                persona.get("persona_id") if isinstance(persona, Mapping) else ""
            )
            if not episode_id or not persona_id:
                raise ValueError("frozen seed rows require episode_id and persona_id")
            expected_personas[episode_id] = persona_id
        source["logs"] = (
            logs_relative if logs.resolve() == expected_logs else logs.name
        )
        source["seeds"] = seeds_relative
        source["seeds_sha256"] = sha256_file(seeds)
        source["dataset_manifest_content_sha256"] = dataset_manifest[
            "manifest_content_sha256"
        ]
    report = build_frozen_report(
        rows,
        split=split,
        require_paired=require_paired,
        source=source,
    )
    if expected_personas is not None:
        if any(_split(row) != split for row in rows):
            raise ValueError("manifest-bound frozen logs must contain only the requested split")
        observed_episode_ids = set(report["episode_ids"])
        if observed_episode_ids != set(expected_personas):
            raise ValueError("frozen evaluation episode IDs do not match the split seeds")
        for row in rows:
            episode_id = _episode_id(row)
            persona_id = str(row.get("persona_id") or "")
            if expected_personas.get(episode_id) != persona_id:
                raise ValueError(
                    f"frozen evaluation persona mismatch for episode {episode_id!r}"
                )
    root = Path(output_dir)
    write_json_atomic(root / "report.json", report)
    _write_text_atomic(root / "report.md", render_report_markdown(report))
    manifest = {
        "schema_version": "feed-evaluation-manifest-v1",
        "source": source,
        "config": {"split": split, "paired": bool(require_paired), "llm_judge": False},
        "files": {
            name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in ("report.json", "report.md")
        },
    }
    manifest["manifest_content_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    write_json_atomic(root / "evaluation_manifest.json", manifest)
    return report


def _verify_frozen_run_manifest(
    run_manifest: str | Path | None,
    *,
    logs: Path,
    split: str,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if run_manifest is None:
        raise ValueError(
            "model rollout logs require a sealed --run-manifest bound to frozen seeds"
        )
    path = Path(run_manifest)
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen rollout run manifest is missing or invalid") from exc
    if run.get("schema_version") != "feed-frozen-rollout-run-v1":
        raise ValueError("frozen rollout run manifest schema mismatch")
    declared = run.get("manifest_content_sha256")
    unsigned = dict(run)
    unsigned.pop("manifest_content_sha256", None)
    computed = sha256_bytes(canonical_json(unsigned).encode("utf-8"))
    if declared != computed:
        raise ValueError("frozen rollout run manifest content hash mismatch")
    if run.get("split") != split:
        raise ValueError("frozen rollout run manifest split mismatch")
    if run.get("dataset_manifest_content_sha256") != dataset_manifest.get(
        "manifest_content_sha256"
    ):
        raise ValueError("frozen rollout run manifest targets another dataset")
    declared_seeds = dataset_manifest.get("files", {}).get(f"seeds/{split}.jsonl")
    if (
        not isinstance(declared_seeds, Mapping)
        or run.get("seeds_sha256") != declared_seeds.get("sha256")
    ):
        raise ValueError("frozen rollout run manifest seed hash mismatch")
    if run.get("logs_sha256") != sha256_file(logs):
        raise ValueError("model rollout logs do not match their run manifest")
    if not str(run.get("policy_id") or ""):
        raise ValueError("frozen rollout run manifest lacks policy_id")
    checkpoint = run.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("sha256"):
        raise ValueError("frozen rollout run manifest lacks checkpoint identity")
    return run


__all__ = [
    "REPORT_METRICS",
    "REPORT_SCHEMA_VERSION",
    "build_frozen_report",
    "evaluate_log_file",
    "render_report_markdown",
]
