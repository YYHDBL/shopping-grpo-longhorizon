# 画像配对复核材料生成与校验
# 用法：
#   python scripts/build_persona_review.py export <shard_count>
#   python scripts/build_persona_review.py check
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.persona.render import render_persona  # noqa: E402

REVIEW = ROOT / "outputs/persona_review"


def load_inputs():
    data = json.load(gzip.open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    instruction = {
        str(r["asin"]): (r["instructions"][0] or {}).get("instruction") or ""
        for r in data
    }
    pool = {}
    for line in (ROOT / "outputs/persona/persona_pool.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        pool[row["persona_id"]] = row
    tasks = [
        json.loads(line)
        for line in (ROOT / "outputs/split/tasks.jsonl").open(encoding="utf-8")
    ]
    return instruction, pool, tasks


def export(shard_count):
    REVIEW.mkdir(parents=True, exist_ok=True)
    instruction, pool, tasks = load_inputs()
    rows = []
    for task in tasks:
        if task["persona_condition"] not in {"aligned", "irrelevant"}:
            continue
        if not task.get("persona_ref"):
            continue
        entry = pool[task["persona_ref"]]
        rows.append({
            "record_id": task["record_id"],
            "condition": task["persona_condition"],
            "instruction": instruction[task["record_id"]],
            "persona_text": render_persona(entry["persona"]),
        })
    shards = [[] for _ in range(shard_count)]
    for index, row in enumerate(rows):
        shards[index % shard_count].append(row)
    for number, batch in enumerate(shards, start=1):
        path = REVIEW / f"shard_{number}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in batch:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(batch)} 条")
    print(f"总计 {len(rows)} 条")


def check():
    expected = set()
    for path in sorted(REVIEW.glob("shard_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            expected.add(json.loads(line)["record_id"])
    got = {}
    for path in sorted(REVIEW.glob("result_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                row = json.loads(line)
                got[row["record_id"]] = row
    missing = expected - set(got)
    from collections import Counter
    counter = Counter((row.get("condition"), row.get("pass")) for row in got.values())
    print(f"期望 {len(expected)}，已判 {len(got)}，缺失 {len(missing)}")
    for (condition, passed), count in sorted(counter.items(), key=str):
        print(f"  {condition} pass={passed}: {count}")


if __name__ == "__main__":
    {"export": lambda: export(int(sys.argv[2])), "check": check}[sys.argv[1]]()
