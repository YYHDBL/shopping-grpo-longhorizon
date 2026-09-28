# 双 Teacher 对比试点：抽 20 条新任务（每难度×条件格 2 条），同格内 1 条给
# GLM-5.3-Flash（Anthropic 协议）、1 条给 DeepSeek V4.1 Flash（OpenAI 协议）。
# 记录：轨迹判定、步数、reward、usage 累计（input/output/cached）。
import gzip
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402
from shopping_grpo.environment.client import ShopAgentEnv  # noqa: E402
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    SYSTEM_PROMPT,
    OpenAIChatClient,
    collect_for_task,
)
from shopping_grpo.persona.render import render_persona  # noqa: E402

OUT = ROOT / "outputs/teacher_compare"
FIRST_PILOT_IDS = {
    json.load(open(ROOT / f"outputs/teacher_pilot/trajectory_{n:02d}.json"))["meta"]["record_id"]
    for n in range(1, 11)
}


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


def pick_pairs(tasks, records_by_id):
    cells = {}
    for task in tasks:
        if task["split"] != "train" or task["record_id"] in FIRST_PILOT_IDS:
            continue
        cells.setdefault((task["difficulty"], task["persona_condition"]), []).append(task)
    pairs = []
    for key in sorted(cells):
        pool = sorted(
            cells[key],
            key=lambda t: len(records_by_id[t["record_id"]]["instructions"][0]["instruction"]),
        )
        middle = len(pool) // 2
        pairs.append((pool[middle - 1], pool[middle + 1]))
        if len(pairs) == 10:
            break
    return pairs


def usage_add(total, usage):
    if not isinstance(usage, dict):
        return total
    total["prompt"] += usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    total["completion"] += usage.get("completion_tokens") or usage.get("output_tokens") or 0
    details = usage.get("prompt_tokens_details") or {}
    total["cached"] += details.get("cached_tokens") or 0
    total["cache_read"] += usage.get("cache_read_input_tokens") or 0
    return total


def run_teacher(name, client, task, record, persona_text, index_of, usage_total):
    system = SYSTEM_PROMPT
    if persona_text:
        system = (SYSTEM_PROMPT + "\n\n以下是该用户的画像，属于软偏好参考，"
                  "与当前请求冲突时以请求为准：\n\n" + persona_text)
    env_task = {
        "task_id": index_of[task["record_id"]],
        "prompt": [{"role": "system", "content": system}],
    }
    trajectory = collect_for_task(
        env_task, client=client, base_url="http://127.0.0.1:5700", max_steps=30)
    usage_add(usage_total, getattr(client, "last_usage", None))
    terminal = trajectory.get("terminal_result") or {}
    reward_detail = terminal.get("reward_detail") or {}
    trajectory["meta"] = {
        "teacher": name,
        "record_id": task["record_id"],
        "difficulty": task["difficulty"],
        "persona_condition": task["persona_condition"],
        "reward_type": reward_detail.get("reward_type"),
        "steps": len(trajectory.get("steps") or []),
        "status": trajectory.get("status"),
        "final_reward": trajectory.get("final_reward"),
    }
    return trajectory


def main():
    load_env_file(ROOT / ".env")
    data = json.load(gzip.open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    index_of = {str(r["asin"]): i for i, r in enumerate(data)}
    records_by_id = {str(r["asin"]): r for r in data}
    tasks = [json.loads(l) for l in (ROOT / "outputs/split/tasks_final.jsonl").open()]
    pool = {json.loads(l)["persona_id"]: json.loads(l)
            for l in (ROOT / "outputs/persona/persona_pool.jsonl").open()}
    pairs = pick_pairs(tasks, records_by_id)
    OUT.mkdir(parents=True, exist_ok=True)

    glm = AnthropicTeacherClient(
        api_key=os.environ["TEACHER_API_KEY"],
        base_url=os.environ["TEACHER_BASE_URL"],
        model=os.environ["TEACHER_MODEL"],
    )
    deepseek = OpenAIChatClient(
        model=os.environ["TEACHER2_MODEL"],
        base_url=os.environ["TEACHER2_BASE_URL"],
        api_key=os.environ["TEACHER2_API_KEY"],
        temperature=1.0,
        top_p=0.95,
        max_tokens=2048,
        timeout=120,
        thinking=False,
    )
    deepseek.extra_headers = {"x-opencode-session": str(uuid.uuid4())}

    usage = {
        "glm": {"prompt": 0, "completion": 0, "cached": 0, "cache_read": 0},
        "deepseek": {"prompt": 0, "completion": 0, "cached": 0, "cache_read": 0},
    }
    number = 0
    for task_a, task_b in pairs:
        for name, client, task in (
            ("glm", glm, task_a), ("deepseek", deepseek, task_b),
        ):
            number += 1
            record = records_by_id[task["record_id"]]
            persona_text = ""
            if task.get("persona_ref"):
                persona_text = render_persona(pool[task["persona_ref"]]["persona"])
            try:
                trajectory = run_teacher(
                    name, client, task, record, persona_text, index_of, usage[name])
            except Exception as exc:
                trajectory = {"meta": {"teacher": name, "record_id": task["record_id"],
                                       "status": f"error:{exc.__class__.__name__}"},
                              "steps": [], "final_reward": None}
            with (OUT / f"trajectory_{number:02d}_{name}.json").open(
                    "w", encoding="utf-8") as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=1)
            print(json.dumps(trajectory.get("meta"), ensure_ascii=False), flush=True)

    print("\nusage 汇总:", json.dumps(usage, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
