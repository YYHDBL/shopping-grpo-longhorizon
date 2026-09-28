# Jev 偏严复查：分片导出 / 覆盖校验与汇总
# 用法：
#   python scripts/jev_review.py export <shard_count>   # 生成等分分片
#   python scripts/jev_review.py check                   # 校验结果覆盖并汇总
#   python scripts/jev_review.py anchor                  # 用 50 条盲判做锚点对比
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.pipeline import load_records, build_jev_state  # noqa: E402

PILOT = ROOT / "outputs/acceptance"
REVIEW = ROOT / "outputs/review"


def _semantic_fail_ids():
    done = set()
    for line in (PILOT / "jev_verdicts.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("ok") and row.get("choice") in {
            "partially_satisfies", "does_not_satisfy"
        }:
            done.add(row["record_id"])
    return done


def export(shard_count):
    REVIEW.mkdir(parents=True, exist_ok=True)
    ids = _semantic_fail_ids()
    records = [r for r in load_records(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    ) if str(r["asin"]) in ids]
    shards = [[] for _ in range(shard_count)]
    for index, record in enumerate(records):
        shards[index % shard_count].append(record)
    for number, batch in enumerate(shards, start=1):
        path = REVIEW / f"shard_{number}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in batch:
                f.write(json.dumps({
                    "record_id": str(record["asin"]),
                    "state": build_jev_state(record),
                }, ensure_ascii=False) + "\n")
        print(f"{path}: {len(batch)} 条")


def check():
    expected = set()
    for path in sorted(REVIEW.glob("shard_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            expected.add(json.loads(line)["record_id"])
    got = {}
    for path in sorted(REVIEW.glob("result_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            got[row["record_id"]] = row
    missing = expected - set(got)
    from collections import Counter
    counter = Counter(v.get("choice") for v in got.values())
    print(f"期望 {len(expected)} 条，已判 {len(got)} 条，缺失 {len(missing)} 条")
    print(f"判定分布: {dict(counter)}")
    if missing:
        (REVIEW / "missing.jsonl").write_text(
            "\n".join(sorted(missing)), encoding="utf-8")
        print(f"缺失清单 -> {REVIEW / 'missing.jsonl'}")
    rescue = sum(1 for v in got.values() if v.get("choice") == "fully_satisfies")
    print(f"复查救回（复查判 fully）: {rescue} 条")


def anchor():
    audit = {}
    result_path = ROOT / "outputs/acceptance-pilot/audit_result.jsonl"
    for line in result_path.open(encoding="utf-8"):
        row = json.loads(line)
        audit[row["record_id"]] = row["choice"]
    got = {}
    for path in sorted(REVIEW.glob("result_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                row = json.loads(line)
                got[row["record_id"]] = row.get("choice")
    overlap = [(rid, audit[rid], got.get(rid)) for rid in audit if rid in got]
    agree = sum(1 for _, a, g in overlap if a == g)
    print(f"锚点重叠 {len(overlap)} 条，与盲判一致 {agree} 条"
          f"（{agree / len(overlap):.0%}）" if overlap else "无重叠")
    for rid, a, g in overlap:
        if a != g:
            print(f"  分歧 {rid}: 盲判={a} 复查={g}")


if __name__ == "__main__":
    {"export": lambda: export(int(sys.argv[2])),
     "check": check, "anchor": anchor}[sys.argv[1]]()
