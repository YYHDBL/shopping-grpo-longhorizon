# Jev 试跑核查：导出盲判材料 / 汇总对比结果
# 用法：
#   python scripts/audit_jev_pilot.py export  # 生成盲判材料
#   python scripts/audit_jev_pilot.py compare  # 对比盲判与 Jev 判定
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.pipeline import load_records, build_jev_state  # noqa: E402

PILOT_DIR = ROOT / "outputs/acceptance-pilot"
MATERIAL = PILOT_DIR / "audit_material.jsonl"
RESULT = PILOT_DIR / "audit_result.jsonl"
VERDICTS = PILOT_DIR / "jev_verdicts.jsonl"


def export():
    records = load_records(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    )[:50]
    with MATERIAL.open("w", encoding="utf-8") as f:
        for record in records:
            row = {
                "record_id": str(record["asin"]),
                "state": build_jev_state(record),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"导出 {len(records)} 条盲判材料 -> {MATERIAL}")


def compare():
    jev = {}
    with VERDICTS.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("ok"):
                jev[row["record_id"]] = row["choice"]
    audit = {}
    with RESULT.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audit[row["record_id"]] = row
    agree = disagree = 0
    conflicts = []
    for record_id, audit_row in audit.items():
        jev_choice = jev.get(record_id)
        if jev_choice == audit_row.get("choice"):
            agree += 1
        else:
            disagree += 1
            conflicts.append({
                "record_id": record_id,
                "jev": jev_choice,
                "audit": audit_row.get("choice"),
                "audit_reason": audit_row.get("reason"),
            })
    total = agree + disagree
    print(f"核查 {total} 条：一致 {agree}，分歧 {disagree}")
    if total:
        print(f"一致率 {agree / total:.1%}")
    for conflict in conflicts:
        print(json.dumps(conflict, ensure_ascii=False))


if __name__ == "__main__":
    {"export": export, "compare": compare}[sys.argv[1]]()
