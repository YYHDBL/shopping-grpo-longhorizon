# L3 案例深读：用 LLM（默认 DeepSeek V4.1 Flash）对比同题上两个模型的轨迹，
# 归纳行为差异与结果归因。抽样自 gain/loss 象限（配对转移的翻转题）。
# 用法: python scripts/deep_read_l3.py --baseline sft-v2-run1 --candidate grpo-run3-step50 \
#          --per-quadrant 8 --workers 4
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics  # noqa: E402
from shopping_grpo.evaluation.rollout import OpenAIChatClient  # noqa: E402
from shopping_grpo.evaluation.trajectory import normalize_trajectory  # noqa: E402

PROMPT = """你是购物 Agent 行为分析专家。同一道购物任务上，两个模型（A=SFT 起点，B=GRPO 强化学习后）各执行了一条轨迹。

任务需求：{instruction}

模型 A 结果：{outcome_a}（{steps_a} 步）
轨迹 A 的动作序列：
{trace_a}

模型 B 结果：{outcome_b}（{steps_b} 步）
轨迹 B 的动作序列：
{trace_b}

请对比两条轨迹并分析。只输出一个 JSON 对象（不要 markdown 代码块），字段：
- "key_differences": 字符串数组，2-4 条关键行为差异（搜索策略/候选选择/核验/终止时机等），每条一句话
- "why_outcome_differs": 一句话，解释结果差异的关键原因
- "pattern": 一句话，概括这条题对反映的行为模式（如"B 更早放弃低价值候选"）"""


def compress_trace(raw: dict, max_obs_chars: int = 160) -> tuple[str, str, int]:
    """轨迹压缩为动作序列文本。返回 (文本, instruction, 步数)。"""
    lines = []
    instruction = ""
    steps = 0
    for m in raw.get("messages", []):
        role = m.get("role")
        if role == "user" and not instruction:
            instruction = str(m.get("content", "")).replace("Instruction: ", "").strip()
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = str(fn.get("arguments", ""))
                if fn.get("name") == "search_products":
                    try:
                        q = json.loads(args).get("query", "")
                        lines.append(f"{steps+1}. 搜索: {q}")
                    except Exception:
                        lines.append(f"{steps+1}. 搜索: {args[:80]}")
                elif fn.get("name") in ("click", "open_product"):
                    lines.append(f"{steps+1}. 打开: {args[:100]}")
                else:
                    lines.append(f"{steps+1}. {fn.get('name')}: {args[:100]}")
                steps += 1
        elif role == "tool":
            obs = str(m.get("content", "")).replace("\n", " ")[:max_obs_chars]
            lines.append(f"   观察: {obs}")
    return "\n".join(lines), instruction, steps


def load_pair(name: str, task_ids: set[int]) -> dict[int, dict]:
    path = ROOT / "outputs/evaluation" / name / "trajectories.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        raw = json.loads(line)
        tid = int(normalize_trajectory(raw)["task_id"])
        if tid in task_ids:
            out[tid] = raw
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="sft-v2-run1")
    parser.add_argument("--candidate", default="grpo-run3-step50")
    parser.add_argument("--per-quadrant", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试）")
    args = parser.parse_args()

    out_dir = ROOT / "outputs/analysis" / f"{args.baseline}_vs_{args.candidate}"
    pairing = json.loads((out_dir / "pairing.json").read_text())
    gain = pairing["quadrants"]["gain(base_fail)"]["task_ids"][: args.per_quadrant]
    loss = pairing["quadrants"]["loss(base_ok)"]["task_ids"][: args.per_quadrant]
    todo = [("gain", t) for t in gain] + [("loss", t) for t in loss]
    if args.limit:
        todo = todo[: args.limit]
    print(f"深读 {len(todo)} 题（gain {len(gain)} + loss {len(loss)}），并发 {args.workers}")

    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()

    client = OpenAIChatClient(
        model=env["TEACHER2_MODEL"],
        base_url=env["TEACHER2_BASE_URL"],
        api_key=env["TEACHER2_API_KEY"],
        temperature=0.0,
        max_tokens=2048,
        timeout=180,
    )
    # opencode 端点强制要求会话 header（缺失时 400；采集脚本同款配置）
    import uuid

    client.extra_headers = {"x-opencode-session": str(uuid.uuid4())}
    lock = threading.Lock()

    base = load_pair(args.baseline, {t for _, t in todo})
    cand = load_pair(args.candidate, {t for _, t in todo})

    def one(item):
        quad, tid = item
        raw_a, raw_b = base[tid], cand[tid]
        trace_a, instruction, steps_a = compress_trace(raw_a)
        trace_b, _, steps_b = compress_trace(raw_b)
        outcome_a = (raw_a.get("terminal_result") or {})
        outcome_b = (raw_b.get("terminal_result") or {})
        ra = normalize_trajectory(raw_a)
        rb = normalize_trajectory(raw_b)
        ma = compute_deterministic_metrics(ra)["reward_and_outcome"]
        mb = compute_deterministic_metrics(rb)["reward_and_outcome"]
        prompt = PROMPT.format(
            instruction=instruction[:300],
            outcome_a=ma.get("reward_type"), steps_a=steps_a, trace_a=trace_a,
            outcome_b=mb.get("reward_type"), steps_b=steps_b, trace_b=trace_b,
        )
        try:
            reply = client.complete([{"role": "user", "content": prompt}], [])
            # complete() 返回 {"role": "assistant", "content": "..."} 字典
            if isinstance(reply, dict):
                text = str(reply.get("content", ""))
            else:
                text = reply if isinstance(reply, str) else getattr(reply, "content", str(reply))
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            analysis = json.loads(text)
        except Exception as exc:
            analysis = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        with lock:
            print(f"  完成 {quad} task_id={tid}")
        return {"quadrant": quad, "task_id": tid, "instruction": instruction[:200],
                "a": {"reward_type": ma.get("reward_type"), "steps": steps_a},
                "b": {"reward_type": mb.get("reward_type"), "steps": steps_b},
                "analysis": analysis}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, item) for item in todo]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda r: (r["quadrant"], r["task_id"]))
    out_path = out_dir / "l3_analyses.jsonl"
    with out_path.open("w", encoding="utf-8") as sink:
        for r in results:
            sink.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总报告
    ok = [r for r in results if "error" not in r["analysis"]]
    lines = [f"# L3 深读报告：{args.baseline} vs {args.candidate}", "",
             f"- 抽样 {len(results)} 题（gain {len(gain)} / loss {len(loss)}），成功 {len(ok)}",
             f"- 评估模型：{env['TEACHER2_MODEL']}", ""]
    for quad in ("gain", "loss"):
        lines.append(f"## {quad} 象限")
        lines.append("")
        for r in [x for x in ok if x["quadrant"] == quad]:
            a = r["analysis"]
            lines.append(f"### task {r['task_id']}（A: {r['a']['reward_type']} → B: {r['b']['reward_type']}）")
            lines.append(f"- 需求: {r['instruction'][:80]}")
            for d in a.get("key_differences", []):
                lines.append(f"- 差异: {d}")
            lines.append(f"- 归因: {a.get('why_outcome_differs', '')}")
            lines.append(f"- 模式: {a.get('pattern', '')}")
            lines.append("")
    (out_dir / "l3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"完成 {len(ok)}/{len(results)}，报告: {out_dir}/l3_report.md")


if __name__ == "__main__":
    main()
