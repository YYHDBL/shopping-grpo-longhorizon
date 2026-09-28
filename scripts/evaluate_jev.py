# 对已完成的评测轨迹执行 V2 Jev 判定、汇总与校准预标。
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import JevDecisionsClient  # noqa: E402
from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402
from shopping_grpo.evaluation.jev import (  # noqa: E402
    ALTERNATIVE_REWARD_TYPES,
    build_calibration_cases,
    build_calibration_report,
    build_case,
    run_glm_prelabels,
    run_jev_cases,
    score_trajectories,
)


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metadata():
    products_path = (
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    )
    with gzip.open(products_path, "rt", encoding="utf-8") as source:
        index_of = {str(row["asin"]): index for index, row in enumerate(json.load(source))}
    tasks_path = ROOT / "outputs/split/tasks_final.jsonl"
    tasks = load_jsonl(tasks_path)
    return {
        index_of[task["record_id"]]: task
        for task in tasks
        if task.get("split") == "eval" and task["record_id"] in index_of
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--glm-workers", type=int, default=2)
    parser.add_argument("--gold-calibration", type=int, default=150)
    parser.add_argument("--grey-calibration", type=int, default=150)
    parser.add_argument("--review-target", type=int, default=100)
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    required = ["OPENROUTER_API_KEY", "TEACHER_API_KEY", "TEACHER_BASE_URL", "TEACHER_MODEL"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit(f"缺少环境变量：{', '.join(missing)}")

    out_dir = ROOT / "outputs/evaluation" / args.name
    trajectories_path = out_dir / "trajectories.jsonl"
    trajectories = load_jsonl(trajectories_path)
    metadata = load_metadata()
    missing_metadata = sorted({int(row["task_id"]) for row in trajectories} - set(metadata))
    if missing_metadata:
        raise ValueError(f"缺少评测任务元数据：{missing_metadata[:10]}")

    evaluation_cases = []
    for trajectory in trajectories:
        detail = (trajectory.get("terminal_result") or {}).get("reward_detail") or {}
        if detail.get("reward_type") in ALTERNATIVE_REWARD_TYPES:
            evaluation_cases.append(build_case(trajectory, metadata[int(trajectory["task_id"])]))
    calibration_cases = build_calibration_cases(
        trajectories,
        metadata,
        gold_count=args.gold_calibration,
        grey_count=args.grey_calibration,
    )

    jev_client = JevDecisionsClient(api_key=os.environ["OPENROUTER_API_KEY"])

    def jev_progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"Jev {done}/{total}", flush=True)

    verdicts = run_jev_cases(
        evaluation_cases + calibration_cases,
        jev_client,
        out_dir / "jev_verdicts.jsonl",
        workers=args.workers,
        progress=jev_progress,
    )
    cases_by_task = {case["task_id"]: case for case in evaluation_cases}
    task_rows, summary = score_trajectories(
        trajectories,
        metadata,
        verdicts,
        cases_by_task,
    )
    write_jsonl(out_dir / "jev_task_results.jsonl", task_rows)
    (out_dir / "jev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(out_dir / "calibration_cases.jsonl", calibration_cases)

    def make_glm_client():
        return AnthropicTeacherClient(
            api_key=os.environ["TEACHER_API_KEY"],
            base_url=os.environ["TEACHER_BASE_URL"],
            model=os.environ["TEACHER_MODEL"],
            temperature=0.0,
            top_p=1.0,
            max_tokens=8192,
            timeout=120,
        )

    def glm_progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"GLM 预标 {done}/{total}", flush=True)

    glm = run_glm_prelabels(
        calibration_cases,
        make_glm_client,
        out_dir / "calibration_glm.jsonl",
        workers=args.glm_workers,
        progress=glm_progress,
    )
    review_path = out_dir / "calibration_human_review.jsonl"
    previous_review = {
        row["calibration_id"]: row for row in load_jsonl(review_path)
    } if review_path.exists() else {}
    audit, review, calibration_report = build_calibration_report(
        calibration_cases,
        verdicts,
        glm,
        review_target=args.review_target,
        human_by_id=previous_review,
    )
    write_jsonl(out_dir / "calibration_audit.jsonl", audit)
    write_jsonl(review_path, review)
    (out_dir / "calibration_report.json").write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps(calibration_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
