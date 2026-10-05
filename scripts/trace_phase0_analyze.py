# TRACE Phase 0 分析：读 credits_*.jsonl，回答三个问题。
# 用法: python scripts/trace_phase0_analyze.py --credits outputs/trace_phase0/credits_run3_50.jsonl
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")

EXPLORE = {"open_product", "view_description", "view_features", "view_reviews",
           "view_attributes", "next_page", "prev_page", "search_products"}
TERMINAL = {"buy_now", "finish_without_purchase"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credits", required=True)
    args = parser.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.credits).open(encoding="utf-8")]
    print(f"轨迹数: {len(rows)}\n")

    # ============ 问题 ①：探索动作的 credit 分布 ============
    print("=" * 60)
    print("① 各动作类型的 turn credit 分布（正=该步让 gold 更可定位）")
    by_action = defaultdict(list)
    for r in rows:
        for action, credit in zip(r["actions"], r["credits"]):
            by_action[action].append(credit)
    print(f"{'动作':<22} {'n':>5} {'均值':>8} {'中位':>8} {'正比例':>7}")
    for action in sorted(by_action, key=lambda a: -len(by_action[a])):
        vals = by_action[action]
        pos = sum(1 for v in vals if v > 0) / len(vals)
        print(f"{action:<22} {len(vals):>5} {statistics.mean(vals):>8.3f} "
              f"{statistics.median(vals):>8.3f} {pos:>6.0%}")

    # δ（不含 terminal fill 的纯近邻信号）——用 mean_target_log_probs 重算
    print("\n--- δ（相邻 turn 的 log-prob 变化，纯探索信号）---")
    by_action_delta = defaultdict(list)
    for r in rows:
        scores = r["mean_target_log_probs"]
        for a, s0, s1 in zip(r["actions"], scores[:-1], scores[1:]):
            by_action_delta[a].append(s1 - s0)  # log-prob 升 = gold 更好预测 = 正贡献
    print(f"{'动作':<22} {'n':>5} {'δ均值':>9} {'δ中位':>9} {'δ>0 比例':>8}")
    for action in sorted(by_action_delta, key=lambda a: -len(by_action_delta[a])):
        vals = by_action_delta[action]
        pos = sum(1 for v in vals if v > 0) / len(vals)
        print(f"{action:<22} {len(vals):>5} {statistics.mean(vals):>9.4f} "
              f"{statistics.median(vals):>9.4f} {pos:>7.0%}")

    # ============ 问题 ②：替代品冲突 ============
    print("\n" + "=" * 60)
    print("② 按结果分组的 credit 特征（替代品冲突检验）")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["reward_type"]].append(r)
    print(f"{'reward_type':<32} {'n':>5} {'尾部credit均值':>13} {'整轨credit和':>12} {'V(S_T)-V(S_0)':>14}")
    for rt in sorted(by_type, key=lambda t: -len(by_type[t])):
        rs = by_type[rt]
        tail = []       # 最后一个 turn 的 credit
        total = []      # 全轨 credit 和
        vspan = []      # log-prob 的总体变化（V 跨度）
        for r in rs:
            if r["credits"]:
                tail.append(r["credits"][-1])
                total.append(sum(r["credits"]))
            s = r["mean_target_log_probs"]
            if len(s) >= 2:
                vspan.append(s[-1] - s[0])
        fmt = lambda xs: f"{statistics.mean(xs):>13.3f}" if xs else f"{'—':>13}"
        print(f"{rt:<32} {len(rs):>5} {fmt(tail)} {fmt(total)} {fmt(vspan)}")

    # ============ 相关性：V 跨度 vs 结果 ============
    print("\n--- gold log-prob 全程变化（S_T - S_0）按结果分组 ---")
    for rt in ("gold_purchase", "partial_alternative_purchase",
               "valid_alternative_purchase", "wrong_purchase", "repeat_loop", "max_steps"):
        rs = by_type.get(rt, [])
        spans = [r["mean_target_log_probs"][-1] - r["mean_target_log_probs"][0]
                 for r in rs if len(r["mean_target_log_probs"]) >= 2]
        if spans:
            print(f"  {rt:<32} n={len(spans):>3} 均值 {statistics.mean(spans):+.4f} "
                  f"中位 {statistics.median(spans):+.4f}")

    # 简易相关（成功 vs 非成功的 V 跨度差异 = 论文的 proxy 有效性检验）
    succ = [r["mean_target_log_probs"][-1] - r["mean_target_log_probs"][0]
            for r in rows if r["reward_type"] == "gold_purchase"
            and len(r["mean_target_log_probs"]) >= 2]
    fail = [r["mean_target_log_probs"][-1] - r["mean_target_log_probs"][0]
            for r in rows if r["reward_type"] not in ("gold_purchase",)
            and len(r["mean_target_log_probs"]) >= 2]
    if succ and fail:
        print(f"\n  proxy 判别力: 成功组 V 跨度均值 {statistics.mean(succ):+.4f} vs "
              f"非成功组 {statistics.mean(fail):+.4f}（差值越大 proxy 越有效）")


if __name__ == "__main__":
    main()
