# SFT 教材加工：accepted 轨迹 -> 规范化训练样本
# 每条样本：Student 版 system（含画像）+ 原消息序列 + 逐消息 loss mask
# mask 规则：assistant 消息参与训练（1），system/user/tool（含守卫回敬）全部遮蔽（0）
# 超长阈值默认不丢弃，仅统计分布；阈值确定后用 --max-chars 重跑生效
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
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    return {
        "record_id": trajectory["meta"]["record_id"],
        "teacher": trajectory["meta"].get("teacher"),
        "difficulty": trajectory["meta"].get("difficulty"),
        "persona_condition": trajectory["meta"].get("persona_condition"),
        "steps": trajectory["meta"].get("steps"),
        "chars": total_chars,
        "messages": messages,
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
    print(f"教材文件 -> {out_path}")
    dist = Counter(s["difficulty"] for s in samples)
    cond = Counter(s["persona_condition"] for s in samples)
    print("难度:", dict(dist), " 条件:", dict(cond))
    train_tokens = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or [], ensure_ascii=False))
                       for s in samples for m in s["messages"] if m["train"])
    print(f"参与训练的动作文本共 {train_tokens:,} 字符")


if __name__ == "__main__":
    main()
