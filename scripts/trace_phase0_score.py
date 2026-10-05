# TRACE Phase 0 离线验证：在已有评测轨迹上计算 turn-level credit，回答三个问题：
# ① 有效探索动作（open_product / view_* / search）后是否得正 credit？
# ② 替代购买（partial/valid_alternative）轨迹上的 credit 是否系统性为负（与 reward 多解冲突）？
# ③ 评分成本（wall time / token 量 / 峰值显存）实测。
# 用法（容器内）:
#   python scripts/trace_phase0_score.py --trajectories outputs/evaluation/grpo-run3-step50/trajectories.jsonl \
#       --targets outputs/trace_phase0/targets.jsonl --model outputs/models/sft-merged \
#       --per-type 30 --out outputs/trace_phase0/credits_run3_50.jsonl
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.grpo.trace import canonical_purchase_target  # noqa: E402

# reward v4 分值表（离线 outcome_advantage 近似；只影响 terminal fill 项）
REWARD_BY_TYPE = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_alternative_purchase": 0.25,
    "graceful_stop": -0.15,
    "early_abstain": -0.35,
    "max_steps": -0.65,
    "repeat_loop": -0.65,
    "wrong_purchase": -1.0,
}

EXPLORE_TOOLS = {"open_product", "view_description", "view_features", "view_reviews",
                 "view_attributes", "next_page", "prev_page", "search_products"}


def _fix_tool_args(messages):
    """渲染前把 function.arguments 从 JSON 字符串解析成 dict（模板用 |items 遍历）。"""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except (TypeError, ValueError):
                        pass
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def turn_prefixes(messages):
    """把 messages 切成 (prefix_messages 列表, 每 turn 的动作名列表)。

    prefix_k = [system, user] + 前 k 组 (assistant, tool) 消息。
    返回 prefixes=[prefix_0, ..., prefix_K] 与 actions=[a_1..a_K]（第 k 个动作对应 δ_k）。
    """
    head = [m for m in messages[:2]]
    prefixes = [head]
    actions = []
    i = 2
    cur = list(head)
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            tool_name = m["tool_calls"][0]["function"]["name"]
            cur = cur + [m]
            actions.append(tool_name)
            # 附上紧随的 tool 返回（渲染时不带 content 细节也 OK——用完整消息更真实）
            if i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                cur = cur + [messages[i + 1]]
                i += 2
            else:
                i += 1
            prefixes.append(list(cur))
        else:
            i += 1
    # prefixes[0] 是初始态，prefixes[k] 对应第 k 个动作之后 → δ_k 对应 actions[k-1]
    return prefixes, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--targets", default="outputs/trace_phase0/targets.jsonl")
    parser.add_argument("--model", default="outputs/models/sft-merged")
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ---- 载入 target 表 ----
    targets = {}
    for line in (ROOT / args.targets).open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("target"):
            targets[row["task_id"]] = row["target"]
    print(f"targets: {len(targets)}")

    # ---- 载入轨迹（按 reward_type 分层抽样；reward_type 经归一化管线提取）----
    from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
    from shopping_grpo.evaluation.trajectory import normalize_trajectory

    by_type = defaultdict(list)
    for line in (ROOT / args.trajectories).open(encoding="utf-8"):
        r = json.loads(line)
        if r["task_id"] not in targets:
            continue
        outcome = compute_deterministic_metrics(normalize_trajectory(r))["reward_and_outcome"]
        r["_reward_type"] = str(outcome.get("reward_type") or "unknown")
        by_type[r["_reward_type"]].append(r)
    sampled = []
    for rt, rows in sorted(by_type.items()):
        take = rows[: args.per_type]
        sampled.extend(take)
        print(f"  {rt}: {len(rows)} 条中抽 {len(take)}")
    print(f"共评 {len(sampled)} 条轨迹")

    # ---- 载入 frozen ref ----
    print("加载模型（frozen ref）...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / args.model), torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True)
    model.eval()
    print(f"模型加载完成 {time.time()-t0:.0f}s | 峰值显存 {torch.cuda.max_memory_allocated()/2**30:.1f}G")

    results = []
    stats = {"forward_calls": 0, "target_tokens": 0, "prefix_tokens": 0}
    t_start = time.time()

    for idx, r in enumerate(sampled):
        task_id = r["task_id"]
        tgt = targets[task_id]
        target_text = canonical_purchase_target(tgt["asin"], tgt.get("options") or [])
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        prefixes, actions = turn_prefixes(_fix_tool_args(r["messages"]))
        # 初始 prefix 的渲染 token（一次算好）
        p0 = tokenizer.apply_chat_template(prefixes[0], tokenize=True,
                                           add_generation_prompt=True)
        if hasattr(p0, "input_ids"):
            p0 = p0.input_ids
        scores = []
        for k, pmsgs in enumerate(prefixes):
            if k == 0:
                ptoks = p0
            else:
                ptoks = tokenizer.apply_chat_template(pmsgs, tokenize=True,
                                                      add_generation_prompt=True)
                if hasattr(ptoks, "input_ids"):
                    ptoks = ptoks.input_ids
            budget = args.max_seq_len - len(target_ids)
            if len(ptoks) > budget:
                ptoks = ptoks[-budget:]  # 保留尾部（最近状态）
            input_ids = torch.tensor([ptoks + target_ids], device=model.device)
            n = len(ptoks)
            with torch.no_grad():
                logits = model(input_ids).logits[0]  # [L, V]
                # 先切出 target 行再做 log_softmax（避免在 24k×152k 全量上算）
                rows = torch.arange(n - 1, n - 1 + len(target_ids), device=logits.device)
                tgt_logits = logits[rows]  # [T, V]
                lp = torch.log_softmax(tgt_logits.float(), dim=-1)
                tgt_lp = lp[torch.arange(len(target_ids), device=logits.device),
                            input_ids[0, n:]]
            scores.append(float(tgt_lp.mean()))
            del logits, tgt_logits, lp
            stats["forward_calls"] += 1
            stats["target_tokens"] += len(target_ids)
            stats["prefix_tokens"] += len(ptoks)
        # ---- TRACE credit ----
        from shopping_grpo.training.grpo.trace import turn_rewards
        rt = r.get("_reward_type") or "unknown"
        out_adv = REWARD_BY_TYPE.get(rt, 0.0)
        credits = turn_rewards(scores, outcome_advantage=out_adv, epsilon=0.1,
                               horizon=3, discount=0.8, terminal_weight=2.0)
        results.append({
            "task_id": task_id, "reward_type": rt,
            "actions": actions,
            "mean_target_log_probs": scores,
            "credits": credits,
            "outcome": out_adv,
            "n_turns": len(actions),
        })
        if (idx + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  {idx+1}/{len(sampled)} | {el:.0f}s | calls {stats['forward_calls']} | "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}G")

    with (ROOT / args.out).open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    wall = time.time() - t_start
    print(f"\n完成 | wall {wall:.0f}s | 前向 {stats['forward_calls']} 次 | "
          f"prefix tokens {stats['prefix_tokens']} | target tokens {stats['target_tokens']} | "
          f"峰值显存 {torch.cuda.max_memory_allocated()/2**30:.1f}G")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
