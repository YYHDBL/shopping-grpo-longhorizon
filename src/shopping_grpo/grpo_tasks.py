"""GRPO 的任务隔离、冻结评测与按实际 rollout 长度分层。"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


LENGTH_BUCKETS = ("short", "medium", "long")
ELIGIBLE_PROBE_STATUSES = frozenset({"done", "max_steps", "assistant_final", "invalid_action_limit"})


def read_jsonl(path: str | Path) -> list[dict]:
    """读取普通或 gzip 压缩的 JSONL；不接受静默损坏行。"""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def task_ids(rows: Iterable[dict]) -> set[int]:
    """从 task 或 trajectory 行提取 task_id，缺失字段立即报错。"""
    ids = set()
    for row in rows:
        if "task_id" not in row:
            raise ValueError("row is missing task_id")
        ids.add(int(row["task_id"]))
    return ids


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_grpo_candidate_manifest(
    all_task_ids: Iterable[int],
    excluded_rollout_task_ids: Iterable[int],
    benchmark_task_ids: Iterable[int],
    size: int,
    seed: int,
) -> list[dict]:
    """从未见题中确定性地挑 probe 候选池。

    排除的是 *raw rollout* 的 task，而非仅 accepted SFT；这样 Teacher 曾经探索过的
    失败题也不会进入在线 RL，避免任务级泄漏的争议。
    """
    excluded = {int(task_id) for task_id in excluded_rollout_task_ids}
    benchmark = {int(task_id) for task_id in benchmark_task_ids}
    candidates = sorted({int(task_id) for task_id in all_task_ids} - excluded - benchmark)
    if size < 1:
        raise ValueError("candidate size must be positive")
    if size > len(candidates):
        raise ValueError("candidate size exceeds available held-out tasks")
    random.Random(int(seed)).shuffle(candidates)
    return [{"task_id": task_id} for task_id in candidates[: int(size)]]


def length_bucket(step_count: int) -> str:
    """使用当前 SFT policy 的实际执行工具步数，而不是 Teacher 轨迹长度。"""
    steps = int(step_count)
    if steps <= 10:
        return "short"
    if steps <= 20:
        return "medium"
    return "long"


def select_stratified_grpo_tasks(
    candidate_rows: Iterable[dict],
    probe_trajectories: Iterable[dict],
    bucket_targets: dict[str, int],
    seed: int,
) -> tuple[list[dict], dict]:
    """根据 probe 的真实步数，按 short/medium/long 精确抽取训练 task。

    基础设施错误没有描述任务难度，故不进入任何桶；任一桶不足即失败，避免把“不平衡”
    悄悄写成看似正式的 GRPO 清单。
    """
    targets = {name: int(bucket_targets.get(name, 0)) for name in LENGTH_BUCKETS}
    if any(count < 0 for count in targets.values()) or not any(targets.values()):
        raise ValueError("bucket targets must contain at least one non-negative positive count")
    candidate_ids = task_ids(candidate_rows)
    grouped: dict[str, list[dict]] = {name: [] for name in LENGTH_BUCKETS}
    ignored = Counter()
    seen = set()
    for trajectory in probe_trajectories:
        task_id = int(trajectory.get("task_id", -1))
        if task_id not in candidate_ids or task_id in seen:
            continue
        seen.add(task_id)
        status = str(trajectory.get("status", "unknown"))
        if status not in ELIGIBLE_PROBE_STATUSES:
            ignored[status] += 1
            continue
        steps = len(trajectory.get("steps") or [])
        bucket = length_bucket(steps)
        grouped[bucket].append({"task_id": task_id, "probe_steps": steps, "length_bucket": bucket})

    available = {name: len(grouped[name]) for name in LENGTH_BUCKETS}
    missing = {name: targets[name] - available[name] for name in LENGTH_BUCKETS if available[name] < targets[name]}
    if missing:
        text = ", ".join(f"{name}: missing {count}" for name, count in sorted(missing.items()))
        raise ValueError(f"insufficient eligible probe tasks by length bucket: {text}")

    selected = []
    for name in LENGTH_BUCKETS:
        rows = list(grouped[name])
        random.Random(f"{seed}:{name}").shuffle(rows)
        selected.extend(rows[: targets[name]])
    random.Random(f"{seed}:final-order").shuffle(selected)
    report = {
        "candidate_task_count": len(candidate_ids),
        "probed_candidate_count": len(seen),
        "eligible_probe_count": sum(available.values()),
        "ignored_probe_status_counts": dict(sorted(ignored.items())),
        "bucket_targets": targets,
        "bucket_available": available,
        "selected_count": len(selected),
        "selected_by_bucket": dict(Counter(row["length_bucket"] for row in selected)),
    }
    return selected, report


def select_disjoint_validation_tasks(
    candidate_rows: Iterable[dict], train_rows: Iterable[dict], size: int, seed: int
) -> list[dict]:
    """从同一冻结候选池中选择未进入在线训练的 validation task。"""
    train = task_ids(train_rows)
    available = sorted(task_ids(candidate_rows) - train)
    if size < 1:
        raise ValueError("validation size must be positive")
    if size > len(available):
        raise ValueError("validation size exceeds tasks remaining after train exclusion")
    random.Random(int(seed)).shuffle(available)
    return [{"task_id": task_id} for task_id in available[: int(size)]]


def freeze_benchmark_subset(
    parent_rows: Iterable[dict], parent_name: str, size: int, parent_sha256: str
) -> tuple[list[dict], dict]:
    """冻结既有 benchmark 的有序前缀，绝不重新随机抽样。"""
    parent = [{"task_id": int(row["task_id"])} for row in parent_rows]
    ids = [row["task_id"] for row in parent]
    if len(set(ids)) != len(ids):
        raise ValueError("parent benchmark contains duplicate task_id")
    if size < 1:
        raise ValueError("subset size must be positive")
    if size > len(parent):
        raise ValueError("subset size exceeds parent benchmark")
    rows = parent[: int(size)]
    return rows, {
        "parent_benchmark": parent_name,
        "parent_sha256": parent_sha256,
        "parent_task_count": len(parent),
        "task_count": len(rows),
        "selection": "ordered_prefix",
    }
