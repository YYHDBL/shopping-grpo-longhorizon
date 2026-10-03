# 轨迹对比分析管线：任意两批评测轨迹的配对对比（同题 join）
# 用法: python scripts/compare_trajectories.py --baseline sft-v2-run1 --candidate grpo-run3-step50
# 口径复用评测管线（normalize_trajectory + compute_deterministic_metrics），不另造判断逻辑。
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics  # noqa: E402
from shopping_grpo.evaluation.trajectory import normalize_trajectory  # noqa: E402

# 行为指标提取清单（字段全来自既有确定性指标，口径与评测一致）
BEHAVIOR_FIELDS = {
    "actions": [
        "executed_tool_steps", "action_attempts", "search_count", "open_product_count",
        "opened_candidate_count", "information_page_count", "select_option_count",
        "buy_count", "finish_without_purchase_count",
    ],
    "repetition": [
        "duplicate_search_query_count", "duplicate_canonical_action_count",
        "consecutive_duplicate_action_count",
    ],
    "legality": [
        "guard_rejection_count", "malformed_tool_call_count", "step_error_count",
    ],
}


def load_arm(name: str) -> dict[int, dict]:
    """加载一批评测，返回 {task_id: 指标包}。"""
    path = ROOT / "outputs/evaluation" / name / "trajectories.jsonl"
    if not path.exists():
        raise SystemExit(f"轨迹文件不存在: {path}")
    arm = {}
    for line in path.open(encoding="utf-8"):
        raw = json.loads(line)
        normalized = normalize_trajectory(raw)
        metrics = compute_deterministic_metrics(normalized)
        outcome = metrics["reward_and_outcome"]
        task_id = int(normalized["task_id"])
        arm[task_id] = {
            "status": raw.get("status"),
            "reward_type": outcome.get("reward_type"),
            "strict": bool(outcome.get("strict_gold_success")),
            "repeat_loop": bool(outcome.get("repeat_loop_termination")),
            "max_steps": bool(outcome.get("max_steps_termination")),
            "infra": bool(raw.get("status") == "error"
                          and (raw.get("error") or {}).get("type") == "HTTPError"
                          or False),
            "actions": {k: metrics["actions_and_efficiency"].get(k, 0)
                        for k in BEHAVIOR_FIELDS["actions"]},
            "repetition": {k: metrics["repetition"].get(k, 0)
                           for k in BEHAVIOR_FIELDS["repetition"]},
            "legality": {k: metrics["legality"].get(k, 0)
                         for k in BEHAVIOR_FIELDS["legality"]},
            "guard_reasons": metrics["legality"].get("guard_reason_counts", {}),
        }
    return arm


def behavior_stats(arm: dict[int, dict], task_ids: list[int]) -> dict:
    """对一组 task_id 聚合行为指标（均值/中位数）。"""
    out = {}
    for group in ("actions", "repetition", "legality"):
        out[group] = {}
        for field in BEHAVIOR_FIELDS[group]:
            vals = [arm[t][group][field] for t in task_ids if t in arm]
            if not vals:
                out[group][field] = None
                continue
            out[group][field] = {
                "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 1),
                "sum": sum(vals),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--sample-per-quadrant", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = load_arm(args.baseline)
    cand = load_arm(args.candidate)
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise SystemExit("两批轨迹没有共同 task_id")
    print(f"配对题数: {len(shared)}（baseline {len(base)} / candidate {len(cand)}）")

    # ---- L1 配对转移 ----
    quadrants = {"both_ok": [], "both_fail": [], "gain(base_fail)": [], "loss(base_ok)": []}
    for t in shared:
        b, c = base[t]["strict"], cand[t]["strict"]
        key = ("both_ok" if b and c else "both_fail" if not b and not c
               else "gain(base_fail)" if c else "loss(base_ok)")
        quadrants[key].append(t)

    n = len(shared)
    b_ok = sum(base[t]["strict"] for t in shared)
    c_ok = sum(cand[t]["strict"] for t in shared)
    pairing = {
        "n": n,
        "baseline_strict": f"{b_ok}/{n} = {b_ok/n:.1%}",
        "candidate_strict": f"{c_ok}/{n} = {c_ok/n:.1%}",
        "delta_pp": round((c_ok - b_ok) / n * 100, 2),
        "quadrants": {k: {"count": len(v), "task_ids": v} for k, v in quadrants.items()},
    }

    # ---- L2 行为对比（整体 + 分象限）----
    behavior = {
        "overall": {
            "baseline": behavior_stats(base, shared),
            "candidate": behavior_stats(cand, shared),
        },
        "by_quadrant": {},
    }
    for q, tids in quadrants.items():
        if not tids:
            continue
        behavior["by_quadrant"][q] = {
            "n": len(tids),
            "baseline": behavior_stats(base, tids),
            "candidate": behavior_stats(cand, tids),
        }

    # 终止类型对比
    behavior["termination"] = {
        "baseline": dict(Counter(base[t]["reward_type"] for t in shared).most_common()),
        "candidate": dict(Counter(cand[t]["reward_type"] for t in shared).most_common()),
    }

    # ---- 输出 ----
    out_dir = Path(args.out) if args.out else (ROOT / "outputs/analysis" / f"{args.baseline}_vs_{args.candidate}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairing.json").write_text(json.dumps(pairing, ensure_ascii=False, indent=2))
    (out_dir / "behavior.json").write_text(json.dumps(behavior, ensure_ascii=False, indent=2))

    # 案例抽样
    cases = {}
    for q, tids in quadrants.items():
        sample = tids[: args.sample_per_quadrant]
        cases[q] = [
            {
                "task_id": t,
                "baseline": {"reward_type": base[t]["reward_type"],
                             "steps": base[t]["actions"]["executed_tool_steps"],
                             "searches": base[t]["actions"]["search_count"]},
                "candidate": {"reward_type": cand[t]["reward_type"],
                              "steps": cand[t]["actions"]["executed_tool_steps"],
                              "searches": cand[t]["actions"]["search_count"]},
            }
            for t in sample
        ]
    (out_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))

    # 报告
    def fmt_row(label, bs, cs):
        def cell(s):
            return "—" if s is None else f"{s['mean']} / {s['median']}"
        return f"| {label} | {cell(bs)} | {cell(cs)} |"

    lines = [
        f"# 轨迹对比报告：{args.baseline} vs {args.candidate}",
        "",
        f"- 配对题数：{n}",
        f"- 严格完成率：baseline {pairing['baseline_strict']} → candidate {pairing['candidate_strict']}"
        f"（Δ {pairing['delta_pp']:+.2f}pp）",
        "",
        "## 配对转移",
        "",
        "| 象限 | 题数 |",
        "|---|---:|",
    ]
    for q in ("both_ok", "both_fail", "gain(base_fail)", "loss(base_ok)"):
        lines.append(f"| {q} | {len(quadrants[q])} |")

    lines += ["", "## 行为指标（均值 / 中位数，全体配对题）", "",
              "| 指标 | baseline | candidate |", "|---|---|---|"]
    for group in ("actions", "repetition", "legality"):
        for field in BEHAVIOR_FIELDS[group]:
            lines.append(fmt_row(field,
                                 behavior["overall"]["baseline"][group][field],
                                 behavior["overall"]["candidate"][group][field]))

    lines += ["", "## 终止类型分布", "",
              f"- baseline: {behavior['termination']['baseline']}",
              f"- candidate: {behavior['termination']['candidate']}"]

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已落盘: {out_dir}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
