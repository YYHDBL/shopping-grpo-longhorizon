# 数据切分与画像分配：难度标注、三条件分配（50/25/25）、aligned/irrelevant 画像初配、dev 划分
# 语义复核（aligned 相关性 / irrelevant 无关性）由 subagent 复核分片完成，本脚本产出初配结果。
# 输出：
#   outputs/persona/persona_pool.jsonl   画像池（persona_id + 原始字段 + 泄漏标签）
#   outputs/split/tasks.jsonl            任务定义（record_id/tag/难度/条件/画像引用/split）
#   outputs/split/report.md              分配统计
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.difficulty import difficulty_of  # noqa: E402

DATA = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
RNG = random.Random(20260925)


def search_keyword_leak(persona, instruction):
    beh = (persona.get("行为特征") or {})
    kws = beh.get("最近14天搜索关键词") or []
    for kw in kws:
        kw = str(kw).strip()
        if len(kw) >= 6 and kw in instruction:
            return True
        if len(kw) >= 6:
            windows = {kw[i:i + 3] for i in range(len(kw) - 2)}
            overlap = sum(1 for w in windows if w in instruction) / len(windows)
            if overlap > 0.6:
                return True
    return False


def build_persona_pool(data):
    pool = []
    for index, record in enumerate(data):
        persona = record.get("user_persona")
        if not isinstance(persona, dict) or not persona:
            continue
        instruction = (record["instructions"][0] or {}).get("instruction") or ""
        pool.append({
            "persona_id": f"P{index:05d}",
            "source_record": str(record["asin"]),
            "domain": record.get("domain_zh"),
            "leak_search": search_keyword_leak(persona, instruction),
            "persona": persona,
        })
    return pool


def assign_conditions(records, ratio=(0.50, 0.25, 0.25)):
    # 难度 × 领域格子内按比例随机分配三条件
    cells = defaultdict(list)
    for record in records:
        cells[(difficulty_of(record), record["domain_zh"])].append(record)
    assignment = {}
    for key, members in cells.items():
        RNG.shuffle(members)
        n = len(members)
        n_aligned = round(n * ratio[1])
        n_irrelevant = round(n * ratio[2])
        n_no = n - n_aligned - n_irrelevant
        for record, condition in zip(
            members,
            ["no_profile"] * n_no
            + ["aligned"] * n_aligned
            + ["irrelevant"] * n_irrelevant,
        ):
            assignment[str(record["asin"])] = condition
    return assignment


def pair_aligned(record, own_persona_entry, pool_by_domain):
    if own_persona_entry is not None and not own_persona_entry["leak_search"]:
        return own_persona_entry["persona_id"], "own"
    candidates = pool_by_domain.get(record["domain_zh"], [])
    if not candidates:
        return None, "unmatched"
    pick = RNG.choice(candidates)
    return pick["persona_id"], "pool"


def pair_irrelevant(record, pool_by_domain):
    other_domains = [d for d in pool_by_domain if d != record["domain_zh"]]
    if not other_domains:
        return None, "unmatched"
    domain = RNG.choice(other_domains)
    pick = RNG.choice(pool_by_domain[domain])
    return pick["persona_id"], f"cross:{domain}"


def main():
    data = json.load(gzip.open(DATA, "rt", encoding="utf-8"))
    final = {}
    for line in (ROOT / "outputs/acceptance/final_verdicts.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        final[row["record_id"]] = row["final_verdict"]
    accepted = [r for r in data if final.get(str(r["asin"])) == "accepted"]

    pool = build_persona_pool(data)
    clean_pool = [p for p in pool if not p["leak_search"]]
    pool_by_domain = defaultdict(list)
    for entry in clean_pool:
        pool_by_domain[entry["domain"]].append(entry)
    own_persona = {p["source_record"]: p for p in pool}
    persona_out = ROOT / "outputs/persona"
    persona_out.mkdir(parents=True, exist_ok=True)
    with (persona_out / "persona_pool.jsonl").open("w", encoding="utf-8") as f:
        for entry in pool:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    tasks = []
    for tag_name, members in (
        ("train", [r for r in accepted if r["tag"] == "train"]),
        ("eval", [r for r in accepted if r["tag"] == "eval"]),
    ):
        assignment = assign_conditions(members)
        for record in members:
            record_id = str(record["asin"])
            condition = assignment[record_id]
            persona_ref = None
            persona_source = None
            if condition == "aligned":
                persona_ref, persona_source = pair_aligned(
                    record, own_persona.get(record_id), pool_by_domain)
            elif condition == "irrelevant":
                persona_ref, persona_source = pair_irrelevant(record, pool_by_domain)
            tasks.append({
                "record_id": record_id,
                "tag": tag_name,
                "difficulty": difficulty_of(record),
                "domain": record["domain_zh"],
                "persona_condition": condition,
                "persona_ref": persona_ref,
                "persona_source": persona_source,
            })

    # dev 划分：train 内按 难度 × 领域 × 条件 分层抽样，总量精确 1050（最大余数法）
    train_cells = defaultdict(list)
    for task in tasks:
        if task["tag"] == "train":
            train_cells[
                (task["difficulty"], task["domain"], task["persona_condition"])
            ].append(task)
    DEV_TARGET = 1050
    train_total = sum(len(m) for m in train_cells.values())
    quotas = {}
    remainders = []
    allocated = 0
    for key, members in train_cells.items():
        exact = len(members) * DEV_TARGET / train_total
        quota = min(len(members), int(exact))
        quotas[key] = quota
        allocated += quota
        remainders.append((exact - quota, key, len(members) - quota))
    remainders.sort(reverse=True)
    index = 0
    while allocated < DEV_TARGET and remainders:
        _, key, headroom = remainders[index % len(remainders)]
        if headroom > quotas[key]:
            quotas[key] += 1
            allocated += 1
        index += 1
        if index > 10 * len(remainders):
            break
    dev_ids = set()
    for key, members in train_cells.items():
        RNG.shuffle(members)
        dev_ids.update(t["record_id"] for t in members[: quotas[key]])
    for task in tasks:
        task["split"] = "dev" if task["record_id"] in dev_ids else (
            "train" if task["tag"] == "train" else "eval")

    split_out = ROOT / "outputs/split"
    split_out.mkdir(parents=True, exist_ok=True)
    with (split_out / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    # 统计
    lines = ["# 数据切分与画像分配统计", ""]
    for tag_name in ("train", "eval"):
        subset = [t for t in tasks if t["tag"] == tag_name]
        cond = Counter(t["persona_condition"] for t in subset)
        diff = Counter(t["difficulty"] for t in subset)
        lines.append(f"- {tag_name}: {len(subset)} 条；条件 {dict(cond)}；难度 {dict(diff)}")
    aligned = [t for t in tasks if t["persona_condition"] == "aligned"]
    src = Counter(t["persona_source"] for t in aligned)
    lines.append(f"- aligned 画像来源: {dict(src)}")
    irrel_missing = sum(1 for t in tasks if t["persona_condition"] == "irrelevant" and not t["persona_ref"])
    lines.append(f"- irrelevant 未配到: {irrel_missing}")
    dev_count = sum(1 for t in tasks if t["split"] == "dev")
    lines.append(f"- dev: {dev_count} 条")
    lines.append(f"- 画像池: {len(pool)} 份（泄漏标记 {sum(p['leak_search'] for p in pool)}，clean {len(clean_pool)}）")
    report = "\n".join(lines) + "\n"
    (split_out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
