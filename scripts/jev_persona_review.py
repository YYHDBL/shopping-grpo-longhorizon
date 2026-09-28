# 用 Jev decisions API 复核画像配对相关性（接管 subagent 未完成的部分）
# 输出与 agent 复查同构：{"record_id","condition","pass","reason"}
#   condition=aligned    related   -> pass=true
#   condition=irrelevant unrelated -> pass=true
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import (  # noqa: E402
    JevApiError,
    JevDecisionsClient,
)

REVIEW = ROOT / "outputs/persona_review"
OUT = REVIEW / "jev_result.jsonl"


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


def remaining_rows():
    rows = {}
    for path in sorted(REVIEW.glob("shard_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            rows[row["record_id"]] = row
    done = set()
    for path in sorted(REVIEW.glob("result_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["record_id"])
            except json.JSONDecodeError:
                continue
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                done.add(row["record_id"])
    return [rows[rid] for rid in rows if rid not in done]


def run(remaining, client, workers=32):
    lock = threading.Lock()
    count = 0
    with OUT.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for row in remaining:
                state = (
                    f"用户购买需求：{row['instruction']}\n\n{row['persona_text']}"
                )
                futures[pool.submit(client.decide_relevance, state)] = row
            for future in as_completed(futures):
                row = futures[future]
                try:
                    verdict = future.result()
                    verdict["condition"] = row["condition"]
                except JevApiError as exc:
                    verdict = {
                        "ok": False,
                        "choice": None,
                        "condition": row["condition"],
                        "error": str(exc)[:200],
                    }
                verdict["record_id"] = row["record_id"]
                with lock:
                    sink.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                    sink.flush()
                    count += 1
                    if count % 200 == 0:
                        print(f"jev 复核进度 {count}/{len(remaining)}", flush=True)


def finalize():
    passed = []
    for line in OUT.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        condition = row.get("condition") or "aligned"
        if not row.get("ok"):
            passed.append({
                "record_id": row["record_id"], "condition": condition,
                "pass": None, "reason": (row.get("error") or "")[:30],
            })
            continue
        choice = row["choice"]
        ok = (condition == "aligned" and choice == "related") or (
            condition == "irrelevant" and choice == "unrelated"
        )
        passed.append({
            "record_id": row["record_id"], "condition": condition,
            "pass": ok, "reason": f"jev:{choice} conf={row.get('confidence')}",
        })
    final_path = REVIEW / "jev_final.jsonl"
    with final_path.open("w", encoding="utf-8") as f:
        for row in passed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok_count = sum(1 for r in passed if r["pass"] is True)
    fail_count = sum(1 for r in passed if r["pass"] is False)
    err_count = sum(1 for r in passed if r["pass"] is None)
    print(f"完成 {len(passed)} 条：合格 {ok_count}，不合格 {fail_count}，错误 {err_count}")
    print(f"折算文件 -> {final_path}")


def main():
    load_env_file(ROOT / ".env")
    remaining = remaining_rows()
    print(f"剩余 {len(remaining)} 条交给 Jev")
    if remaining:
        client = JevDecisionsClient(api_key=os.environ["OPENROUTER_API_KEY"])
        run(remaining, client)
    finalize()


if __name__ == "__main__":
    main()
