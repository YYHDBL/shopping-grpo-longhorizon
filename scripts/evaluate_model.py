# 冻结评测入口：对已 serve 的模型在 tag=eval 切分上跑完整轨迹并汇总指标
# Baseline 与正式 Evaluation 共用本入口，区别只在 serve 的模型权重：
#   bash scripts/serve_model.sh models/Qwen3.5-9B          # Baseline：未训练底座
#   bash scripts/serve_model.sh outputs/models/sft-merged  # Evaluation：训练后检查点
# 前置：ShopSimulator 环境已由 scripts/start_environment.sh 启动。
# 用法：
#   python scripts/evaluate_model.py --name baseline
#   python scripts/evaluate_model.py --name sft --limit 50
# 产出：
#   outputs/evaluation/<name>/trajectories.jsonl  逐任务轨迹（断点续跑）
#   outputs/evaluation/<name>/summary.json        严格成功等汇总指标
import argparse
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    CollectionInfrastructureError,
    STUDENT_SYSTEM_PROMPT,
    OpenAIChatClient,
    _is_infrastructure_failure,
    append_jsonl,
    collect_for_task,
    completed_task_attempts,
)
from shopping_grpo.evaluation.trajectory import normalize_trajectory  # noqa: E402
from shopping_grpo.persona.render import render_persona  # noqa: E402

DEFAULT_TASKS = ROOT / "outputs/split/tasks_final.jsonl"
ENV_PRODUCT_DATA = (
    ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
)


def build_env_tasks(split, limit):
    tasks = [json.loads(l) for l in DEFAULT_TASKS.open(encoding="utf-8")]
    selected = [t for t in tasks if t.get("split") == split]
    if limit:
        selected = selected[: int(limit)]
    persona_pool = {
        json.loads(l)["persona_id"]: json.loads(l)
        for l in (ROOT / "outputs/persona/persona_pool.jsonl").open(encoding="utf-8")
    }
    with gzip.open(ENV_PRODUCT_DATA, "rt", encoding="utf-8") as f:
        index_of = {str(row["asin"]): i for i, row in enumerate(json.load(f))}
    env_tasks = []
    for task in sorted(selected, key=lambda t: t["record_id"]):
        system = STUDENT_SYSTEM_PROMPT
        if task.get("persona_ref"):
            entry = persona_pool.get(task["persona_ref"])
            if entry:
                system += ("\n\n以下是该用户的画像，属于软偏好参考，"
                           "与当前请求冲突时以请求为准：\n\n"
                           + render_persona(entry["persona"]))
        env_tasks.append({
            "task_id": index_of[task["record_id"]],
            "record_id": task["record_id"],
            "difficulty": task["difficulty"],
            "persona_condition": task["persona_condition"],
            "domain": task["domain"],
            "prompt": [{"role": "system", "content": system}],
        })
    return env_tasks


def summarize(trajectories_path, tasks):
    strict_task_ids = set()
    trajectory_count = 0
    reward_types = Counter()
    for raw in trajectories_path.open(encoding="utf-8"):
        normalized = normalize_trajectory(json.loads(raw))
        outcome = compute_deterministic_metrics(normalized)["reward_and_outcome"]
        if outcome["strict_gold_success"]:
            strict_task_ids.add(int(normalized["task_id"]))
        reward_types[str(outcome["reward_type"])] += 1
        trajectory_count += 1

    dimensions = {
        "difficulty": "by_difficulty",
        "persona_condition": "by_persona_condition",
        "domain": "by_domain",
    }
    buckets = {name: {} for name in dimensions.values()}
    for field, name in dimensions.items():
        totals = Counter(str(task[field]) for task in tasks)
        successes = Counter(
            str(task[field]) for task in tasks if int(task["task_id"]) in strict_task_ids
        )
        buckets[name] = {
            value: {
                "task_count": count,
                "strict_gold_success_count": successes[value],
                "strict_gold_success_rate": successes[value] / count,
            }
            for value, count in sorted(totals.items())
        }

    expected = len(tasks)
    strict = len(strict_task_ids)
    return {
        "expected_task_count": expected,
        "trajectory_count": trajectory_count,
        "strict_gold_success_count": strict,
        "strict_gold_success_rate": (strict / expected) if expected else 0.0,
        "reward_type_counts": dict(sorted(reward_types.items())),
        **buckets,
    }


def collect_tasks_parallel(tasks, make_client, output_path, base_url, max_steps, workers):
    done = completed_task_attempts(output_path)
    pending = [task for task in tasks if (int(task["task_id"]), 0) not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                collect_for_task,
                task,
                client=make_client(),
                base_url=base_url,
                max_steps=max_steps,
            ): task
            for task in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            trajectory = future.result()
            append_jsonl(output_path, [trajectory])
            print(
                f"完成 {len(tasks) - len(pending) + completed}/{len(tasks)}："
                f"task_id={trajectory['task_id']} status={trajectory['status']}",
                flush=True,
            )
            if _is_infrastructure_failure(trajectory):
                for other in futures:
                    other.cancel()
                raise CollectionInfrastructureError(
                    "collection infrastructure failure; stopped before remaining tasks"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="评测名称，作为输出子目录")
    parser.add_argument("--model", default="shopping-agent",
                        help="serve 时的 served-model-name（scripts/serve_model.sh）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--split", default="eval",
                        help="tasks_final.jsonl 的切分标签，默认冻结 eval 集")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试）")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = ROOT / "outputs/evaluation" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = out_dir / "trajectories.jsonl"

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    def make_client():
        return OpenAIChatClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_tokens,
            timeout=120,
            thinking=False,
            context_window=24576,
            context_safety_margin=512,
            context_compaction_enable=False,
            observation_token_budget=1536,
            observation_detail_token_budget=4096,
            observation_generic_token_budget=768,
            observation_search_top_k=20,
        )

    env_tasks = build_env_tasks(args.split, args.limit)
    print(f"评测 {args.name}：{len(env_tasks)} 题，模型 {args.base_url} 的 "
          f"{args.model}，环境 {args.env_url}", flush=True)
    collect_tasks_parallel(
        env_tasks,
        make_client=make_client,
        output_path=trajectories_path,
        base_url=args.env_url,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    summary = {
        "schema_version": "shopping-evaluation-summary-v1",
        **summarize(trajectories_path, env_tasks),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
