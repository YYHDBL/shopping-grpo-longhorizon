"""验收管线编排：两级流水、断点续跑、判定合并与汇总报告。

设计基线 4.2（2026-09-23 确认）的实现。输出三个文件：
- level1_checks.jsonl   第一级逐条明细（本地，重跑覆盖）
- jev_verdicts.jsonl    第二级逐条明细（追加，断点续跑按 record_id 跳过）
- report.json / report.md  汇总
"""

from __future__ import annotations

import gzip
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping

from shopping_grpo.acceptance.difficulty import difficulty_of
from shopping_grpo.acceptance.hard_checks import run_hard_checks
from shopping_grpo.acceptance.jev_client import (
    JEV_RUBRIC_VERSION,
    JevApiError,
    JevDecisionsClient,
)

L1_PASS = "pass"
L1_UNVERIFIABLE = "unverifiable"

VERDICTS = ("accepted", "hard_fail", "semantic_fail", "unverifiable")


def load_records(data_path) -> list[dict]:
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_jev_state(record: Mapping) -> str:
    """Jev 只看实际用户需求与 Gold 商品字段；不含画像、轨迹或 DOM。"""
    instructions = record.get("instructions") or []
    instruction = str((instructions[0] or {}).get("instruction") or "")
    required = list((instructions[0] or {}).get("instruction_options") or [])
    pricing = record.get("pricing") or []
    price = pricing[0] if pricing else None
    lines = [
        f"用户需求：{instruction}",
        "候选商品（参考答案 Gold）：",
        f"- 标题：{record.get('title') or ''}",
        f"- 店铺：{record.get('shop_name') or ''}",
        f"- 类目：{record.get('category') or ''}",
        f"- 价格：{price} 元",
        f"- 关键属性：{', '.join(record.get('attribute') or [])}",
    ]
    if required:
        lines.append(f"- 必选规格：{'; '.join(str(v) for v in required)}")
    return "\n".join(lines)


def _read_completed(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_id") and row.get("ok"):
                done.add(row["record_id"])
    return done


def run_level1(records) -> dict[str, dict]:
    return {str(r["asin"]): run_hard_checks(r) for r in records}


def run_level2(
    records: list[dict],
    client: JevDecisionsClient,
    out_path: Path,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """并发调用 Jev；输出追加写入，已完成（ok=true）的 record_id 跳过。"""
    done = _read_completed(out_path)
    queue = [r for r in records if str(r["asin"]) not in done]
    results: dict[str, dict] = {}
    write_lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(client.decide_satisfaction, build_jev_state(r)): r
                for r in queue
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record = futures[future]
                record_id = str(record["asin"])
                try:
                    verdict = future.result()
                except JevApiError as exc:
                    verdict = {
                        "ok": False,
                        "choice": None,
                        "error": str(exc),
                        "request_hash": client.request_hash(
                            build_jev_state(record)
                        ),
                        "rubric_version": JEV_RUBRIC_VERSION,
                    }
                verdict["record_id"] = record_id
                with write_lock:
                    sink.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                    sink.flush()
                results[record_id] = verdict
                if progress and index % 50 == 0:
                    progress(index, len(queue))
    return results


def combine_verdict(level1: Mapping, level2: Mapping | None) -> dict:
    """合并两级结果为最终判定。"""
    if level1.get("status") == "fail":
        failed = [c["name"] for c in level1.get("checks", []) if c["status"] == "fail"]
        return {"verdict": "hard_fail", "reason": f"hard checks failed: {failed}"}
    if level2 is None:
        return {"verdict": "unverifiable", "reason": "level2 not run"}
    if not level2.get("ok"):
        return {"verdict": "unverifiable", "reason": level2.get("error") or "level2 error"}
    choice = level2.get("choice")
    if choice == "fully_satisfies":
        return {"verdict": "accepted", "reason": "gold satisfies the request"}
    if choice in {"partially_satisfies", "does_not_satisfy"}:
        return {"verdict": "semantic_fail", "reason": f"jev: {choice}"}
    return {"verdict": "unverifiable", "reason": f"jev: {choice}"}


def build_report(records, level1: Mapping, level2: Mapping, jev_model_reported=None) -> dict:
    verdicts = {}
    for record in records:
        record_id = str(record["asin"])
        verdicts[record_id] = combine_verdict(
            level1.get(record_id), level2.get(record_id)
        )
    if jev_model_reported is None:
        jev_model_reported = next(
            (
                v.get("model_reported")
                for v in level2.values()
                if v.get("model_reported")
            ),
            None,
        )
    total = len(records)
    verdict_counts = Counter(v["verdict"] for v in verdicts.values())
    by_tag: dict[str, Counter] = {}
    by_difficulty: dict[str, Counter] = {}
    by_domain: dict[str, Counter] = {}
    failure_reasons = Counter()
    for record in records:
        record_id = str(record["asin"])
        verdict = verdicts[record_id]["verdict"]
        for grouping, key in (
            (by_tag, str(record.get("tag"))),
            (by_difficulty, difficulty_of(record)),
            (by_domain, str(record.get("domain_zh"))),
        ):
            grouping.setdefault(key, Counter())[verdict] += 1
        if verdict in {"hard_fail", "semantic_fail"}:
            failure_reasons[verdicts[record_id]["reason"]] += 1
    return {
        "total": total,
        "verdict_counts": dict(verdict_counts),
        "by_tag": {k: dict(v) for k, v in by_tag.items()},
        "by_difficulty": {k: dict(v) for k, v in by_difficulty.items()},
        "by_domain": {k: dict(v) for k, v in by_domain.items()},
        "top_failure_reasons": dict(failure_reasons.most_common(20)),
        "jev_model_reported": jev_model_reported,
        "level2_completed": sum(1 for v in level2.values() if v.get("ok")),
    }


def _report_markdown(report: dict) -> str:
    lines = [
        "# 数据验收报告",
        "",
        f"- 总量：{report['total']}",
        f"- 判定：{json.dumps(report['verdict_counts'], ensure_ascii=False)}",
        f"- Jev 实际版本：{report.get('jev_model_reported')}",
        f"- 二级完成数：{report.get('level2_completed')}",
        "",
        "## 按 tag",
    ]
    for tag, counts in sorted(report["by_tag"].items()):
        lines.append(f"- {tag}: {json.dumps(counts, ensure_ascii=False)}")
    lines.append("")
    lines.append("## 按难度")
    for key, counts in sorted(report["by_difficulty"].items()):
        lines.append(f"- {key}: {json.dumps(counts, ensure_ascii=False)}")
    lines.append("")
    lines.append("## 按领域")
    for key, counts in sorted(report["by_domain"].items()):
        lines.append(f"- {key}: {json.dumps(counts, ensure_ascii=False)}")
    lines.append("")
    lines.append("## 失败原因 Top")
    for reason, count in report["top_failure_reasons"].items():
        lines.append(f"- {count} × {reason}")
    return "\n".join(lines) + "\n"


def run(
    data_path,
    out_dir,
    client: JevDecisionsClient | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """执行完整验收流水；dry_run 只跑第一级。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(data_path)
    if limit:
        records = records[: int(limit)]

    level1 = run_level1(records)
    l1_path = out_dir / "level1_checks.jsonl"
    with l1_path.open("w", encoding="utf-8") as f:
        for record_id, result in level1.items():
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    level2: dict[str, dict] = {}
    if not dry_run and client is not None:
        eligible = [
            r for r in records
            if level1[str(r["asin"])].get("status") in {L1_PASS, L1_UNVERIFIABLE}
        ]
        l2_path = out_dir / "jev_verdicts.jsonl"
        level2 = run_level2(
            eligible, client, l2_path, workers=workers, progress=progress
        )
        # 报告需要全量二级结果（含此前断点续跑累积的）。
        level2.update(_read_all_verdicts(l2_path))

    report = build_report(records, level1, level2)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    return report


def _read_all_verdicts(path: Path) -> dict[str, dict]:
    verdicts: dict[str, dict] = {}
    if not path.exists():
        return verdicts
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_id"):
                verdicts[row["record_id"]] = row
    return verdicts
