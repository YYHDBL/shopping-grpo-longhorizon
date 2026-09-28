# 验收定案：合并两级验收、复查结果与异常定价剔除，产出最终判定
# 规则（2026-09-25 用户确认）：
#   L1 fail                        -> hard_fail
#   Jev fully                      -> accepted
#   Jev partial/does_not:
#     复查 fully 且 Gold 标价 >1 元 -> accepted（复查救回，信复查）
#     复查 fully 且标价 <=1 元      -> bad_pricing（异常定价，确认坏）
#     复查 partial/does_not/insufficient -> semantic_fail
#   不可验证                       -> unverifiable
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPT = ROOT / "outputs/acceptance"
REVIEW = ROOT / "outputs/review"

ANOMALY_PRICE_LIMIT = 1.0


def main():
    data = json.load(gzip.open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    pricing = {str(r["asin"]): r.get("pricing") or [None] for r in data}
    tag = {str(r["asin"]): r.get("tag") for r in data}

    l1 = {}
    for line in (ACCEPT / "level1_checks.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        l1[row["record_id"]] = row["status"]
    jev = {}
    for line in (ACCEPT / "jev_verdicts.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("ok"):
            jev[row["record_id"]] = row["choice"]
    review = {}
    for path in sorted(REVIEW.glob("result_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                row = json.loads(line)
                review[row["record_id"]] = row

    final = {}
    for record_id, status in l1.items():
        if status == "fail":
            final[record_id] = ("hard_fail", "level1")
        elif jev.get(record_id) == "fully_satisfies":
            final[record_id] = ("accepted", "jev")
        elif record_id in review:
            if review[record_id]["choice"] == "fully_satisfies":
                price = pricing[record_id][0]
                if price is not None and float(price) <= ANOMALY_PRICE_LIMIT:
                    final[record_id] = ("bad_pricing", f"gold price {price}")
                else:
                    final[record_id] = ("accepted", "review_rescue")
            else:
                final[record_id] = ("semantic_fail", f"review:{review[record_id]['choice']}")
        elif jev.get(record_id) in {"partially_satisfies", "does_not_satisfy"}:
            final[record_id] = ("semantic_fail", "no_review")
        else:
            final[record_id] = ("unverifiable", "price_or_api")

    out = ACCEPT / "final_verdicts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for record_id, (verdict, source) in final.items():
            f.write(json.dumps({
                "record_id": record_id,
                "final_verdict": verdict,
                "source": source,
                "tag": tag.get(record_id),
            }, ensure_ascii=False) + "\n")

    counter = Counter(v for v, _ in final.values())
    by_tag = {"train": Counter(), "eval": Counter()}
    for record_id, (verdict, _) in final.items():
        by_tag[tag.get(record_id) or "unknown"][verdict] += 1
    print("最终判定:", dict(counter))
    for name, counts in by_tag.items():
        print(f"  {name}: {dict(counts)}")
    print(f"定案文件 -> {out}")


if __name__ == "__main__":
    main()
