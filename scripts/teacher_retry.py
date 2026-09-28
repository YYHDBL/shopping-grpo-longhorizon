# 用新 Teacher 提示词重跑此前失败的任务，验证收敛优化效果
# glm 重跑 1 条死循环；deepseek 重跑 4 条超步 + 1 条死循环
import gzip
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    OpenAIChatClient,
    collect_for_task,
)
from shopping_grpo.persona.render import render_persona  # noqa: E402

OUT = ROOT / "outputs/teacher_retry"

RETRY = {
    "glm": ["925125896291"],
    "deepseek": ["938140389011", "815302416447", "630041182276",
                 "830817437170", "907083131191"],
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


def main():
    load_env_file(ROOT / ".env")
    data = json.load(gzip.open(
        ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
        "rt", encoding="utf-8"))
    index_of = {str(r["asin"]): i for i, r in enumerate(data)}
    records_by_id = {str(r["asin"]): r for r in data}
    tasks = {json.loads(l)["record_id"]: json.loads(l)
             for l in (ROOT / "outputs/split/tasks_final.jsonl").open()}
    pool = {json.loads(l)["persona_id"]: json.loads(l)
            for l in (ROOT / "outputs/persona/persona_pool.jsonl").open()}
    OUT.mkdir(parents=True, exist_ok=True)

    clients = {
        "glm": AnthropicTeacherClient(
            api_key=os.environ["TEACHER_API_KEY"],
            base_url=os.environ["TEACHER_BASE_URL"],
            model=os.environ["TEACHER_MODEL"],
        ),
        "deepseek": OpenAIChatClient(
            model=os.environ["TEACHER2_MODEL"],
            base_url=os.environ["TEACHER2_BASE_URL"],
            api_key=os.environ["TEACHER2_API_KEY"],
            temperature=1.0, top_p=0.95, max_tokens=2048,
            timeout=120, thinking=False,
        ),
    }
    clients["deepseek"].extra_headers = {"x-opencode-session": str(uuid.uuid4())}

    number = 0
    for name, record_ids in RETRY.items():
        for record_id in record_ids:
            number += 1
            task = tasks[record_id]
            system = TEACHER_SYSTEM_PROMPT
            if task.get("persona_ref"):
                system += ("\n\n以下是该用户的画像，属于软偏好参考，"
                           "与当前请求冲突时以请求为准：\n\n"
                           + render_persona(pool[task["persona_ref"]]["persona"]))
            env_task = {"task_id": index_of[record_id],
                        "prompt": [{"role": "system", "content": system}]}
            try:
                trajectory = collect_for_task(
                    env_task, client=clients[name],
                    base_url="http://127.0.0.1:5700", max_steps=30)
            except Exception as exc:
                trajectory = {"status": f"error:{exc.__class__.__name__}"}
            terminal = trajectory.get("terminal_result") or {}
            reward_detail = terminal.get("reward_detail") or {}
            meta = {
                "teacher": name, "record_id": record_id,
                "difficulty": task["difficulty"],
                "condition": task["persona_condition"],
                "reward_type": reward_detail.get("reward_type"),
                "steps": len(trajectory.get("steps") or []),
                "status": trajectory.get("status"),
                "reward": trajectory.get("final_reward"),
            }
            trajectory["meta"] = meta
            with (OUT / f"trajectory_{number:02d}_{name}.json").open(
                    "w", encoding="utf-8") as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=1)
            print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
