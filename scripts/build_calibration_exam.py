# 生成 Jev 校准考卷：从真实判定历史分层抽样，人工复核用
# 池子：采集期替代品判定（fully / 其他）、数据验收 Jev fully、Jev partial、复查分歧救回
import glob
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs/calibration"
RNG = random.Random(20260927)

QUOTA = {
    "alt_fully": 15,      # 采集期替代品判完全满足（已进教材，重点复核）
    "alt_rejected": 5,    # 采集期替代品判部分/不满足（被拒）
    "gold_fully": 12,     # 数据验收判完全满足
    "gold_partial": 13,   # 数据验收判部分满足（偏严重灾区）
    "rescued": 5,         # 复查救回的口径敏感条目
}


def load_data():
    data = json.load(gzip.open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    return {str(r["asin"]): r for r in data}


def gold_product_text(record):
    ins = record["instructions"][0]
    return (f"标题：{record['title']}\n"
            f"价格：{record['pricing'][0]} 元\n"
            f"属性：{'、'.join(record.get('attribute') or [])}\n"
            f"必选规格：{'、'.join(ins.get('instruction_options') or [])}")


def purchased_product_text(terminal):
    purchase = terminal.get("purchase") or {}
    return (f"标题：{purchase.get('name')}\n"
            f"价格：{purchase.get('price')} 元\n"
            f"所选规格：{purchase.get('options')}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_data()
    pools = {key: [] for key in QUOTA}

    # 池 1/2：采集期替代品判定
    for path in glob.glob(str(ROOT / "outputs/collection/trajectories/*.json")):
        t = json.load(open(path))
        meta = t.get("meta") or {}
        reason = str(meta.get("accept_reason") or "")
        if reason == "jev:fully_satisfies" and len(pools["alt_fully"]) < 400:
            terminal = t.get("terminal_result") or {}
            pools["alt_fully"].append({
                "source": "替代品·Jev判完全满足（已进教材）",
                "instruction": (terminal.get("purchase") or {}).get("instruction_text"),
                "product": purchased_product_text(terminal),
                "jev": "fully_satisfies",
            })
        elif reason.startswith("jev:") and reason != "jev:fully_satisfies":
            if len(pools["alt_rejected"]) < 200:
                terminal = t.get("terminal_result") or {}
                pools["alt_rejected"].append({
                    "source": f"替代品·{reason}（被拒）",
                    "instruction": (terminal.get("purchase") or {}).get("instruction_text"),
                    "product": purchased_product_text(terminal),
                    "jev": reason.split(":", 1)[1],
                })

    # 池 3/4/5：数据验收判定与复查救回
    review = {}
    for path in sorted((ROOT / "outputs/review").glob("result_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            review[row["record_id"]] = row
    final = {}
    for line in (ROOT / "outputs/acceptance/final_verdicts.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        final[row["record_id"]] = row
    jev_choice = {}
    for line in (ROOT / "outputs/acceptance/jev_verdicts.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("ok"):
            jev_choice[row["record_id"]] = row["choice"]

    for record_id, verdict in final.items():
        record = records.get(record_id)
        if record is None:
            continue
        instruction = record["instructions"][0]["instruction"]
        base = {
            "source": "",
            "instruction": instruction,
            "product": gold_product_text(record),
            "jev": jev_choice.get(record_id, "?"),
        }
        review_row = review.get(record_id)
        review_choice = review_row.get("choice") if review_row else None
        if verdict["final_verdict"] == "accepted" and verdict["source"] == "review_rescue":
            entry = dict(base)
            entry["source"] = "复查救回（Jev曾判不满足）"
            entry["review"] = "fully_satisfies"
            pools["rescued"].append(entry)
        elif jev_choice.get(record_id) == "fully_satisfies":
            entry = dict(base)
            entry["source"] = "数据验收·Jev判完全满足"
            pools["gold_fully"].append(entry)
        elif jev_choice.get(record_id) == "partially_satisfies" and review_choice != "fully_satisfies":
            entry = dict(base)
            entry["source"] = "数据验收·Jev判部分满足"
            if review_choice:
                entry["review"] = review_choice
            pools["gold_partial"].append(entry)

    lines = ["# Jev 校准考卷（50 条）", "",
             "每题你选一个：fully（完全满足）/ partial（部分满足）/ does_not（不满足）/ unknown（说不清）",
             "把答案写每题末尾\"你的判定：\"后面即可，或者直接在对话里报题号和选项。", ""]
    questions = []
    number = 0
    for key, quota in QUOTA.items():
        pool = pools[key]
        RNG.shuffle(pool)
        for entry in pool[:quota]:
            number += 1
            lines.append(f"## 第 {number} 题")
            lines.append(f"来源：{entry['source']}")
            lines.append(f"用户需求：{entry['instruction']}")
            lines.append(f"商品：{entry['product']}")
            lines.append(f"Jev 判定：{entry['jev']}")
            if entry.get("review"):
                lines.append(f"复查判定：{entry['review']}")
            lines.append("你的判定：")
            lines.append("")
            questions.append({"number": number, "pool": key, **entry})
    (OUT / "exam_50.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "exam_50.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"考卷生成：{number} 题 -> {OUT / 'exam_50.md'}（同结构 json 版 exam_50.json）")
    for key, quota in QUOTA.items():
        print(f"  {key}: 需要 {quota}，池子 {len(pools[key])}")


if __name__ == "__main__":
    main()
