# SFT 教材加工：accepted 轨迹 -> 规范化训练样本
# 每条样本：Student 版 system（含画像）+ 原消息序列 + 逐消息 loss mask
# mask 规则：正常 assistant 消息参与训练（1）；system/user/tool 全部遮蔽（0）；
# 被本地动作守卫拒绝的 assistant 段同样遮蔽（0）——守卫拒绝是应当规避的行为，
# 不进入 loss。守卫拒绝以 tool 消息的 runtime_action_guard 标记 + tool_call_id
# 关联到对应 assistant 段。
# 准入硬性校验（fail fast，不做静默修复）：
#   1) 终止审计完整：reward_valid=true、over=true、终止原因为在；
#   2) 工具调用参数为合法 JSON 对象，且不含 null/空字符串参数值
#      （历史数据里 view_features({"__dummy__": null}) 一类占位参数属于
#      语义污染，含此类调用的整条轨迹直接剔除并计数，禁止静默转换）。
# 每条样本同时落盘准入审计字段（accept_reason/reward_type/reward_valid/over/
# termination_reason/jev_verdict/purchase），替代购买（JEV 准入）的依据可独立追溯。
# 超长阈值默认不丢弃，仅统计分布；真实 token 过滤由 export_sft_parquet.py 完成。
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.environment.actions import RUNTIME_GUARD_FIELD  # noqa: E402
from shopping_grpo.evaluation.rollout import STUDENT_SYSTEM_PROMPT  # noqa: E402
from shopping_grpo.persona.render import render_persona  # noqa: E402

OUT = ROOT / "outputs/sft_dataset"


def tool_call_argument_problem(tool_call):
    """返回非法参数原因；合法返回 None。禁止静默转换，只判定不修复。"""
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw = function.get("arguments")
    if raw is None:
        return f"{name}:arguments_missing"
    if not isinstance(raw, str):
        return f"{name}:arguments_not_json_string"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return f"{name}:arguments_invalid_json"
    if not isinstance(parsed, dict):
        return f"{name}:arguments_not_object"
    for key, value in parsed.items():
        if value is None:
            return f"{name}:argument_value_null:{key}"
        if isinstance(value, str) and not value:
            return f"{name}:argument_value_empty:{key}"
    return None


def admission_problem(trajectory):
    """教材准入校验：终止审计 + 全部工具调用参数。返回原因或 None。"""
    terminal = trajectory.get("terminal_result") or {}
    reward_detail = terminal.get("reward_detail") or {}
    if not terminal or not isinstance(reward_detail, dict) or not reward_detail.get("reward_type"):
        return "terminal_incomplete"
    if terminal.get("over") is not True:
        return "terminal_not_over"
    if terminal.get("reward_valid") is not True or reward_detail.get("reward_valid") is not True:
        return "reward_invalid"
    if not terminal.get("termination_reason"):
        return "terminal_incomplete"
    for message in trajectory.get("messages") or []:
        for tool_call in message.get("tool_calls") or []:
            problem = tool_call_argument_problem(tool_call)
            if problem:
                return f"invalid_arguments:{problem}"
    return None


def admission_audit(trajectory):
    """从轨迹 meta.audit 与终止结果组装样本级审计字段。"""
    meta = trajectory.get("meta") or {}
    audit = meta.get("audit") or {}
    terminal = trajectory.get("terminal_result") or {}
    reward_detail = terminal.get("reward_detail") or {}
    purchase = audit.get("purchase") or terminal.get("purchase") or {}
    return {
        "accept_reason": meta.get("accept_reason"),
        "reward_type": reward_detail.get("reward_type") or meta.get("reward_type"),
        "reward_valid": terminal.get("reward_valid"),
        "over": terminal.get("over"),
        "termination_reason": terminal.get("termination_reason"),
        "jev_verdict": audit.get("jev_verdict"),
        "purchase": (
            {
                "name": purchase.get("name"),
                "price": purchase.get("price"),
                "options": purchase.get("options"),
                "instruction_text": purchase.get("instruction_text"),
            }
            if purchase
            else None
        ),
    }


def build_sample(trajectory, task, persona_pool):
    # system 换 Student 版；画像按任务条件拼入
    system = STUDENT_SYSTEM_PROMPT
    if task.get("persona_ref"):
        entry = persona_pool.get(task["persona_ref"])
        if entry:
            system += ("\n\n以下是该用户的画像，属于软偏好参考，"
                       "与当前请求冲突时以请求为准：\n\n"
                       + render_persona(entry["persona"]))
    messages = []
    original = trajectory.get("messages") or []
    instruction = next(
        (m.get("content") for m in original if m.get("role") == "user"),
        "",
    )
    messages.append({"role": "system", "content": system, "train": 0})
    messages.append({"role": "user", "content": instruction, "train": 0})
    for message in original:
        role = message.get("role")
        if role == "assistant":
            messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
                "train": 1,
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": message.get("tool_call_id"),
                "content": message.get("content"),
                "guard": message.get(RUNTIME_GUARD_FIELD, False),
                "train": 0,
            })
    # 守卫拒绝的 assistant 段不参与训练：以 tool 消息的守卫标记 +
    # tool_call_id 关联被拒绝的调用（同一 id 只可能属于一个 assistant 段）。
    rejected_call_ids = {
        m.get("tool_call_id")
        for m in original
        if m.get(RUNTIME_GUARD_FIELD)
    }
    excluded_segments = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if any(call.get("id") in rejected_call_ids for call in calls):
            message["train"] = 0
            excluded_segments += 1
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    return {
        "record_id": trajectory["meta"]["record_id"],
        "teacher": trajectory["meta"].get("teacher"),
        "difficulty": trajectory["meta"].get("difficulty"),
        "persona_condition": trajectory["meta"].get("persona_condition"),
        "steps": trajectory["meta"].get("steps"),
        "chars": total_chars,
        "guard_excluded_segments": excluded_segments,
        "messages": messages,
        **admission_audit(trajectory),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=None,
                        help="超长丢弃阈值（字符数），默认只统计不丢弃")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import glob
    tasks = {json.loads(l)["record_id"]: json.loads(l)
             for l in (ROOT / "outputs/split/tasks_final.jsonl").open()}
    persona_pool = {json.loads(l)["persona_id"]: json.loads(l)
                    for l in (ROOT / "outputs/persona/persona_pool.jsonl").open()}

    status = {}
    teacher_of = {}
    for name in ("glm", "deepseek"):
        for line in (ROOT / f"outputs/collection/task_status_{name}.jsonl").open():
            row = json.loads(line)
            if row["final"] == "accepted":
                status[row["record_id"]] = row
                teacher_of[row["record_id"]] = name

    samples = []
    seen = set()
    skipped_missing = 0
    rejected_admission = Counter()
    rejected_admission_ids = {}
    for path in sorted(glob.glob(str(ROOT / "outputs/collection/trajectories/*.json"))):
        rid = Path(path).name.rsplit("_", 1)[0]
        if rid not in status or rid in seen:
            continue
        trajectory = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = trajectory.get("meta") or {}
        if not meta.get("accepted"):
            continue
        seen.add(rid)
        task = tasks.get(rid)
        if task is None:
            skipped_missing += 1
            continue
        problem = admission_problem(trajectory)
        if problem is not None:
            rejected_admission[problem] += 1
            rejected_admission_ids.setdefault(problem, []).append(rid)
            continue
        sample = build_sample(trajectory, task, persona_pool)
        sample["teacher"] = teacher_of.get(rid)
        samples.append(sample)

    chars = sorted(s["chars"] for s in samples)
    n = len(chars)
    stats = {
        "samples": n,
        "chars_p50": chars[n // 2],
        "chars_p90": chars[int(n * 0.9)],
        "chars_p95": chars[int(n * 0.95)],
        "chars_p99": chars[int(n * 0.99)],
        "chars_max": chars[-1],
        "skipped_missing_task": skipped_missing,
        "dropped_admission": dict(rejected_admission),
        "guard_excluded_segments": sum(s["guard_excluded_segments"] for s in samples),
        "trajectories_with_guard_exclusion": sum(
            1 for s in samples if s["guard_excluded_segments"]),
        "accept_reasons": dict(Counter(s["accept_reason"] for s in samples)),
    }
    if args.max_chars:
        kept = [s for s in samples if s["chars"] <= args.max_chars]
        stats["dropped_over_limit"] = n - len(kept)
        samples = kept
        stats["kept"] = len(kept)

    out_path = OUT / "samples.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if rejected_admission_ids:
        print("剔除明细 ->", json.dumps(rejected_admission_ids, ensure_ascii=False))
    print(f"教材文件 -> {out_path}")
    dist = Counter(s["difficulty"] for s in samples)
    cond = Counter(s["persona_condition"] for s in samples)
    print("难度:", dict(dist), " 条件:", dict(cond))
    train_tokens = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or [], ensure_ascii=False))
                       for s in samples for m in s["messages"] if m["train"])
    print(f"参与训练的动作文本共 {train_tokens:,} 字符")


if __name__ == "__main__":
    main()
