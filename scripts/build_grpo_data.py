# V2 GRPO 任务数据构建：Student 版 prompt、与 SFT 任务隔离、按难度分层
# 桶配额（用户 2026-09-29 确认）：teacher_hard（双 Teacher 均未做出来的题，SFT 教材缺失，
# RL 探索价值最高）+ hard + medium + easy，难度区分明确且覆盖 SFT 阶段没学好的盲区。
# smoke 模式 --limit 64 只出小批量；正式跑 --limit 1000
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")
sys.path.insert(0, str(ROOT / "src"))

# 正式跑桶配额：th=teacher_hard。比例 30/30/28/12，标签口径下 hard 类约 45%
QUOTA = {"teacher_hard": 0.30, "hard": 0.30, "medium": 0.28, "easy": 0.12}

from shopping_grpo.evaluation.rollout import STUDENT_SYSTEM_PROMPT  # noqa: E402
from shopping_grpo.persona.render import render_persona  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--out", default="outputs/grpo/smoke.parquet")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tasks = [json.loads(l) for l in (ROOT / "outputs/split/tasks_final.jsonl").open()]
    sft_used = set(
        pd.read_parquet(ROOT / "outputs/sft_dataset/train.parquet")["record_id"]
    )
    pool = [
        t for t in tasks
        if t["split"] == "train" and t["record_id"] not in sft_used
    ]
    # teacher_hard 集：采集时被分配的 Teacher 三次尝试均失败（双 Teacher 分治，每题只归一个 Teacher）
    th_ids = set()
    for name in ("glm", "deepseek"):
        with (ROOT / f"outputs/collection/task_status_{name}.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("final") == "teacher_hard":
                    th_ids.add(r["record_id"])
    # 四个桶：teacher_hard 单独成桶（跨难度），其余按难度分桶
    by_diff = defaultdict(list)
    for t in pool:
        key = "teacher_hard" if t["record_id"] in th_ids else t["difficulty"]
        by_diff[key].append(t)
    for members in by_diff.values():
        rng.shuffle(members)
    n = min(args.limit, len(pool))
    quotas = {k: min(len(v), round(n * QUOTA[k])) for k, v in by_diff.items()}
    # 配额不足时把缺口按桶大小摊回其他桶
    deficit = n - sum(quotas.values())
    order = sorted(by_diff, key=lambda k: -len(by_diff[k]))
    i = 0
    while deficit > 0:
        k = order[i % len(order)]
        if quotas[k] < len(by_diff[k]):
            quotas[k] += 1
            deficit -= 1
        i += 1
        if i > 10 * n:  # 全池不够，防死循环
            break
    selected = []
    for k, members in by_diff.items():
        selected.extend(members[: quotas[k]])
    selected_ids = {t["record_id"] for t in selected}

    # 画像渲染
    persona_pool = {}
    for l in (ROOT / "outputs/persona/persona_pool.jsonl").open():
        row = json.loads(l)
        persona_pool[row["persona_id"]] = row
    # 任务行号（环境 task_id）
    import gzip
    data = json.load(
        gzip.open(
            ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
            "rt", encoding="utf-8",
        )
    )
    index_of = {str(r["asin"]): i for i, r in enumerate(data)}
    instr_of = {
        str(r["asin"]): r["instructions"][0]["instruction"] for r in data
    }

    rows = []
    for t in selected:
        system = STUDENT_SYSTEM_PROMPT
        if t.get("persona_ref"):
            entry = persona_pool.get(t["persona_ref"])
            if entry:
                system += (
                    "\n\n以下是该用户的画像，属于软偏好参考，"
                    "与当前请求冲突时以请求为准：\n\n"
                    + render_persona(entry["persona"])
                )
        rows.append({
            "data_source": "shopping_v2",
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Instruction: {instr_of[t['record_id']]}"},
            ],
            "ability": "shopping",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {
                "task_id": index_of[t["record_id"]],
                "record_id": t["record_id"],
                "difficulty": t["difficulty"],
                "persona_condition": t["persona_condition"],
                "teacher_hard": t["record_id"] in th_ids,
            },
        })
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    diff = Counter(r["extra_info"]["difficulty"] for r in rows)
    cond = Counter(r["extra_info"]["persona_condition"] for r in rows)
    th_n = sum(1 for r in rows if r["extra_info"]["teacher_hard"])
    print(f"写出 {len(rows)} 条 -> {out}")
    print(f"难度 {dict(diff)} | 画像 {dict(cond)} | teacher_hard {th_n} | 池总量 {len(pool)}")


if __name__ == "__main__":
    main()
