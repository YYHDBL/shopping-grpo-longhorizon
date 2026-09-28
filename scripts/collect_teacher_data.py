# Teacher 正式采集：成功即停（每题预算 3 次），在线验收，断点续跑，分级落盘
# 用法：
#   python scripts/collect_teacher_data.py --limit 3            # 试跑 3 题
#   python scripts/collect_teacher_data.py --limit 6000         # 正式
#   python scripts/collect_teacher_data.py --status             # 查看进度与统计
# 产出：
#   outputs/collection/trajectories/<record_id>_<attempt>.json  全部轨迹
#   outputs/collection/task_status.jsonl                         每题终态（断点）
#   outputs/collection/report.json                               汇总
import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import (  # noqa: E402
    JevApiError,
    JevDecisionsClient,
)
from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    OpenAIChatClient,
    collect_for_task,
)
from shopping_grpo.persona.render import render_persona  # noqa: E402

OUT = ROOT / "outputs/collection"
TRAJ = OUT / "trajectories"
BUDGET = 3
RNG = random.Random(20260926)
STRUCTURAL_FAIL = {"max_steps", "invalid_action_limit", "error", "model_no_tool_call"}


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


def pick_tasks(limit, shard=0, shards=1):
    tasks = [json.loads(l) for l in (ROOT / "outputs/split/tasks_final.jsonl").open()]
    pool = sorted(
        (t for t in tasks if t["split"] == "train"),
        key=lambda t: t["record_id"],
    )
    # 按 record_id 稳定分片：同一个任务永远落在同一个分片，双 Teacher 不重叠。
    pool = pool[shard::shards]
    done = set()
    for status_path in sorted(OUT.glob("task_status_*.jsonl")):
        for line in status_path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("final") in {"accepted", "teacher_hard", "env_exhausted"}:
                done.add(row["record_id"])
    remaining = [t for t in pool if t["record_id"] not in done]
    # 难度分层：hard 至少 30%，其余近似源分布
    by_diff = defaultdict(list)
    for t in remaining:
        by_diff[t["difficulty"]].append(t)
    for members in by_diff.values():
        RNG.shuffle(members)
    n = min(int(limit), len(remaining))
    n_hard = min(len(by_diff.get("hard", [])), round(n * 0.30))
    n_rest = n - n_hard
    non_hard = by_diff.get("easy", []) + by_diff.get("medium", [])
    source_rest = len(non_hard)
    n_easy = round(n_rest * len(by_diff.get("easy", [])) / max(source_rest, 1))
    selected = (by_diff.get("hard", [])[:n_hard]
                + by_diff.get("easy", [])[:n_easy]
                + by_diff.get("medium", [])[:n_rest - n_easy])
    return selected


def make_client(name):
    load_env_file(ROOT / ".env")
    if name == "glm":
        return AnthropicTeacherClient(
            api_key=os.environ["TEACHER_API_KEY"],
            base_url=os.environ["TEACHER_BASE_URL"],
            model=os.environ["TEACHER_MODEL"],
        )
    client = OpenAIChatClient(
        model=os.environ["TEACHER2_MODEL"],
        base_url=os.environ["TEACHER2_BASE_URL"],
        api_key=os.environ["TEACHER2_API_KEY"],
        temperature=1.0, top_p=0.95, max_tokens=2048,
        timeout=120, thinking=False,
    )
    client.extra_headers = {"x-opencode-session": str(uuid.uuid4())}
    return client


def judge_trajectory(trajectory, jev, record, thread_clients):
    """在线验收：返回 (accepted: bool, reason: str)。"""
    status = trajectory.get("status")
    terminal = trajectory.get("terminal_result") or {}
    reward_detail = terminal.get("reward_detail") or {}
    reward_type = reward_detail.get("reward_type")
    if status in STRUCTURAL_FAIL:
        return False, f"structural:{status}"
    if reward_type == "gold_purchase":
        return True, "gold"
    if reward_type in {"valid_alternative_purchase", "partial_alternative_purchase"}:
        purchase = terminal.get("purchase") or {}
        try:
            verdict = jev.decide_satisfaction(
                f"用户需求：{purchase.get('instruction_text')}\n"
                f"候选商品（Agent 实际购买）：{purchase.get('name')}，"
                f"价格 {purchase.get('price')} 元，"
                f"所选规格 {purchase.get('options')}，"
                f"关键属性 {'、'.join(record.get('attribute') or [])}"
            )
        except JevApiError as exc:
            return False, f"jev_error:{str(exc)[:40]}"
        if verdict["choice"] == "fully_satisfies":
            return True, "jev:fully_satisfies"
        return False, f"jev:{verdict['choice']}"
    return False, f"env:{reward_type or status}"


def run_task(task, index_of, records_by_id, persona_pool, jev, teacher="glm"):
    client = make_client(teacher)
    record = records_by_id[task["record_id"]]
    system = TEACHER_SYSTEM_PROMPT
    if task.get("persona_ref"):
        entry = persona_pool.get(task["persona_ref"])
        if entry:
            system += ("\n\n以下是该用户的画像，属于软偏好参考，"
                       "与当前请求冲突时以请求为准：\n\n"
                       + render_persona(entry["persona"]))
    env_task = {"task_id": index_of[task["record_id"]],
                "prompt": [{"role": "system", "content": system}]}
    attempt = 0
    env_failures = 0
    infra_files = 0
    while attempt < BUDGET:
        try:
            trajectory = collect_for_task(
                env_task, client=client,
                base_url="http://127.0.0.1:5700", max_steps=30)
        except Exception as exc:
            # Teacher/网络层异常属于基础设施故障，不消耗采样预算。
            env_failures += 1
            if env_failures > 12:
                return {"record_id": task["record_id"], "final": "env_exhausted",
                        "attempts": attempt, "usage": dict(client.total_usage)}
            time.sleep(min(60, 2 ** env_failures))
            continue
        error_type = ((trajectory.get("error") or {}).get("type") or "")
        if trajectory.get("status") == "error" and error_type in {
            "ShopEnvironmentError", "ShopHttpError",
        }:
            # 基础设施故障不消耗采样预算；指数退避，最多容忍 12 次。
            env_failures += 1
            if env_failures > 12:
                return {"record_id": task["record_id"], "final": "env_exhausted",
                        "attempts": attempt, "usage": dict(client.total_usage)}
            time.sleep(min(60, 2 ** env_failures))
            continue
        attempt += 1
        accepted, reason = judge_trajectory(trajectory, jev, record, client)
        infra_files += int(
            trajectory.get("status") == "error"
            or reason.startswith("structural:error"))
        trajectory["meta"] = {
            "record_id": task["record_id"],
            "difficulty": task["difficulty"],
            "persona_condition": task["persona_condition"],
            "attempt": attempt,
            "accepted": accepted,
            "accept_reason": reason,
            "reward_type": (trajectory.get("terminal_result") or {})
                .get("reward_detail", {}).get("reward_type"),
            "steps": len(trajectory.get("steps") or []),
            "status": trajectory.get("status"),
        }
        with (TRAJ / f"{task['record_id']}_{attempt}.json").open(
                "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False)
        if accepted:
            return {"record_id": task["record_id"], "final": "accepted",
                    "attempts": attempt, "reason": reason,
                    "usage": dict(client.total_usage)}
    final = ("teacher_hard" if infra_files < attempt else "env_exhausted")
    return {"record_id": task["record_id"], "final": final,
            "attempts": attempt,
            "usage": dict(client.total_usage)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--teacher", default="glm", choices=["glm", "deepseek"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    TRAJ.mkdir(parents=True, exist_ok=True)
    if args.status:
        counts = Counter()
        for status_path in sorted(OUT.glob("task_status_*.jsonl")):
            for line in status_path.open(encoding="utf-8"):
                row = json.loads(line)
                counts[row["final"]] += 1
        print(dict(counts))
        return

    load_env_file(ROOT / ".env")
    data = json.load(gzip.open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    index_of = {str(r["asin"]): i for i, r in enumerate(data)}
    records_by_id = {str(r["asin"]): r for r in data}
    tasks_all = [json.loads(l) for l in (ROOT / "outputs/split/tasks_final.jsonl").open()]
    persona_pool = {json.loads(l)["persona_id"]: json.loads(l)
                    for l in (ROOT / "outputs/persona/persona_pool.jsonl").open()}
    selected = pick_tasks(args.limit, shard=args.shard, shards=args.shards)
    print(f"本轮采集 {len(selected)} 题（预算 {BUDGET} 次/题，成功即停，"
          f"{args.workers} 并发）", flush=True)

    jev = JevDecisionsClient(api_key=os.environ["OPENROUTER_API_KEY"])
    write_lock = threading.Lock()
    done_count = 0
    consecutive_infra = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t, index_of, records_by_id,
                               persona_pool, jev, args.teacher): t
                   for t in selected}
        for future in as_completed(futures):
            result = future.result()
            with write_lock:
                status_path = OUT / f"task_status_{args.teacher}.jsonl"
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                done_count += 1
                if result["final"] == "env_exhausted":
                    consecutive_infra += 1
                else:
                    consecutive_infra = 0
                if done_count % 20 == 0:
                    print(f"进度 {done_count}/{len(selected)}", flush=True)
                # 全局熔断：连续 50 题基础设施失败说明外网或环境长时间不可用，
                # 停止烧队列，恢复后重启续跑。
                if consecutive_infra >= 50:
                    print("连续 50 题基础设施失败，熔断退出。恢复后重启续跑。",
                          flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    counts = Counter()
    usage_total = {"input": 0, "output": 0, "cached": 0}
    for status_path in sorted(OUT.glob("task_status_*.jsonl")):
        for line in status_path.open(encoding="utf-8"):
            row = json.loads(line)
            counts[row["final"]] += 1
            for key in usage_total:
                usage_total[key] += (row.get("usage") or {}).get(key, 0)
    report = {"tasks": dict(counts), "usage_total": usage_total}
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
