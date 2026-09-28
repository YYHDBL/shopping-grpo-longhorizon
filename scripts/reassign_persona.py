# 画像配对重配循环：复核不合格的配对重新抽取画像并用 Jev 复判，最多 3 轮，
# 仍失败的任务降级为 no_profile 并记录。产出最终任务定义文件。
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import JevApiError, JevDecisionsClient  # noqa: E402
from shopping_grpo.persona.render import render_persona  # noqa: E402

REVIEW = ROOT / "outputs/persona_review"
MAX_ROUNDS = 3
RNG = random.Random(20260926)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def load_all():
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
        if not row["leak_search"]:
            pool[row["persona_id"]] = row
    tasks = [
        json.loads(line)
        for line in (ROOT / "outputs/split/tasks.jsonl").open(encoding="utf-8")
    ]
    review = {}
    for i in range(1, 11):
        path = REVIEW / f"result_{i}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            review[row["record_id"]] = row["pass"]
    for line in (REVIEW / "jev_final.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        review.setdefault(row["record_id"], row["pass"])
    return instruction, pool, tasks, review


def pick_persona(condition, task, pool, exclude):
    candidates = [
        p for p in pool.values()
        if p["persona_id"] not in exclude
        and (p["domain"] == task["domain"]
             if condition == "aligned"
             else p["domain"] != task["domain"])
    ]
    if not candidates:
        return None
    return RNG.choice(candidates)


def main():
    load_env_file(ROOT / ".env")
    client = JevDecisionsClient(api_key=os.environ["OPENROUTER_API_KEY"])
    instruction, pool, tasks, review = load_all()
    pool_by_domain = defaultdict(list)
    for entry in pool.values():
        pool_by_domain[entry["domain"]].append(entry)

    # 待重配队列：复核不合格（pass=False）
    pending = []
    demoted = []
    for task in tasks:
        if task["persona_condition"] not in {"aligned", "irrelevant"}:
            continue
        if not task.get("persona_ref"):
            continue
        passed = review.get(task["record_id"])
        if passed is False:
            pending.append(task)
    print(f"待重配 {len(pending)} 条")

    for round_number in range(1, MAX_ROUNDS + 1):
        if not pending:
            break
        verdicts = {}
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {}
            for task in pending:
                entry = pick_persona(
                    task["persona_condition"], task, pool,
                    exclude={task["persona_ref"]},
                )
                if entry is None:
                    verdicts[task["record_id"]] = (None, "no_candidate")
                    continue
                state = (
                    f"用户购买需求：{instruction[task['record_id']]}\n\n"
                    + render_persona(entry["persona"])
                )
                future = executor.submit(client.decide_relevance, state)
                futures[future] = (task, entry)
            for future, (task, entry) in futures.items():
                try:
                    result = future.result()
                    related = result["choice"] == "related"
                    ok = (task["persona_condition"] == "aligned") == related
                    verdicts[task["record_id"]] = (entry["persona_id"] if ok else None,
                                                   "ok" if ok else f"jev:{result['choice']}")
                except JevApiError as exc:
                    verdicts[task["record_id"]] = (None, str(exc)[:60])
        still = []
        for task in pending:
            persona_id, note = verdicts.get(task["record_id"], (None, "missing"))
            if persona_id:
                task["persona_ref"] = persona_id
                task["persona_source"] = f"reassign_r{round_number}"
                review[task["record_id"]] = True
            else:
                still.append(task)
        pending = still
        print(f"第 {round_number} 轮：重配成功 "
              f"{sum(1 for v in verdicts.values() if v[0])}，剩余 {len(pending)}")

    for task in pending:
        task["persona_condition"] = "no_profile"
        task["persona_ref"] = None
        task["persona_source"] = "demoted_from_" + task["persona_source"]
        demoted.append(task["record_id"])
    print(f"降级 no_profile：{len(demoted)} 条")

    out = ROOT / "outputs/split/tasks_final.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    cond = Counter((t["tag"], t["persona_condition"]) for t in tasks)
    print("最终条件分布:")
    for key, count in sorted(cond.items()):
        print(f"  {key}: {count}")
    print(f"最终任务文件 -> {out}")


if __name__ == "__main__":
    main()
