# GRPO 训练数据顺序重排（2026-10-01 用户方案）：
# 常规难度（easy+medium+hard，非 teacher_hard）混合打乱为一个桶，
# teacher_hard 桶放最后（课程学习：常规题热身 → 最难题目收尾）。
# 任务集完全不变，只改行序；写入新文件，原 train_1000.parquet 保留供 run1/run2 追溯。
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/data/jyh-yyh/shopping-grpo-longhorizon")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outputs/grpo/train_1000.parquet")
    parser.add_argument("--out", default="outputs/grpo/train_1000_curriculum.parquet")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    df = pd.read_parquet(ROOT / args.src)
    ex = df["extra_info"].apply(pd.Series)
    th_mask = ex["teacher_hard"].astype(bool)

    regular = df[~th_mask].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    th = df[th_mask].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out = pd.concat([regular, th], ignore_index=True)

    # ---- 自校验（任务集与隔离性必须与源完全一致，只允许顺序变化）----
    src_ids = set(ex["record_id"])
    out_ids = set(out["extra_info"].apply(lambda d: d["record_id"]))
    assert src_ids == out_ids, "任务集发生了变化！"

    tasks = [json.loads(l) for l in (ROOT / "outputs/split/tasks_final.jsonl").open()]
    data = json.load(__import__("gzip").open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    index_of = {str(r["asin"]): i for i, r in enumerate(data)}
    eval_ids = set(index_of[t["record_id"]] for t in tasks
                   if t["split"] == "eval" and t["record_id"] in index_of)
    sft = set(pd.read_parquet(ROOT / "outputs/sft_dataset/train.parquet")["record_id"])
    sft_ids = set(index_of[r] for r in sft if r in index_of)

    ex_out = out["extra_info"].apply(pd.Series)
    out_task_ids = set(ex_out["task_id"])
    assert not (out_task_ids & eval_ids), "与冻结评测集发生重叠！"
    assert not (out_task_ids & sft_ids), "与 SFT 教材发生重叠！"

    th_out = ex_out["teacher_hard"].astype(bool)
    assert not th_out.iloc[: len(regular)].any(), "常规桶里混入了 teacher_hard"
    assert th_out.iloc[len(regular):].all(), "teacher_hard 桶不完整"

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"写出 {len(out)} 条 -> {out_path}")
    print(f"桶结构: [常规混合 {len(regular)} 条（桶内已打乱）] → [teacher_hard {len(th)} 条（已打乱）]")
    print(f"自校验通过: 任务集一致 | ∩eval=0 | ∩SFT=0 | 桶边界正确")
    print("新顺序前 10 条难度:", "".join(d[0].upper() for d in ex_out["difficulty"].iloc[:10]))
    print("新顺序末 10 条难度:", "".join(d[0].upper() for d in ex_out["difficulty"].iloc[-10:]))


if __name__ == "__main__":
    main()
