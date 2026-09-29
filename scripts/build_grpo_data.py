# V2 GRPO 任务数据构建：Student 版 prompt、与 SFT 任务隔离、按难度分层
# smoke 模式 --limit 64 只出小批量；正式跑 --limit 12000
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")
sys.path.insert(0, str(ROOT / "src"))

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
    # 难度分层：hard 30%，其余近似源分布
    by_diff = defaultdict(list)
    for t in pool:
        by_diff[t["difficulty"]].append(t)
    for members in by_diff.values():
        rng.shuffle(members)
    n = min(args.limit, len(pool))
    n_hard = min(len(by_diff.get("hard", [])), round(n * 0.30))
    n_rest = n - n_hard
    non_hard = by_diff.get("easy", []) + by_diff.get("medium", [])
    rng.shuffle(non_hard)
    selected = by_diff.get("hard", [])[:n_hard] + non_hard[:n_rest]

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
            },
        })
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    diff = Counter(r["extra_info"]["difficulty"] for r in rows)
    cond = Counter(r["extra_info"]["persona_condition"] for r in rows)
    print(f"写出 {len(rows)} 条 -> {out}")
    print(f"难度 {dict(diff)} | 画像 {dict(cond)} | 池总量 {len(pool)}")


if __name__ == "__main__":
    main()
