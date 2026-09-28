# Teacher 小规模试点：抽 10 条任务跑轨迹并做 V2 轨迹验收
# 验收 = 环境终判（Reward v3）+ 非 Gold 购买调 Jev 判替代 + 轨迹良构
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import JevDecisionsClient, JevApiError  # noqa: E402
from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402
from shopping_grpo.environment.client import ShopAgentEnv  # noqa: E402
from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT, collect_for_task  # noqa: E402
from shopping_grpo.persona.render import render_persona  # noqa: E402

OUT = ROOT / "outputs/teacher_pilot"
GOOD_TERMINATIONS = {"gold_purchase", "valid_alternative_purchase",
                     "partial_alternative_purchase"}
STRUCTURAL_OK = {"done"}


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


def pick_tasks(tasks, records_by_id):
    cells = {}
    for task in tasks:
        if task["split"] != "train":
            continue
        cells.setdefault((task["difficulty"], task["persona_condition"]), []).append(task)
    picked = []
    for key in sorted(cells):
        pool = cells[key]
        # 每格取第一条指令长度适中的，避免极端长指令干扰试点
        pool = sorted(pool, key=lambda t: len(records_by_id[t["record_id"]]["instructions"][0]["instruction"]))
        picked.append(pool[len(pool) // 2])
        if len(picked) == 9:
            break
    picked.append(cells[("medium", "no_profile")][0])
    return picked[:10]


def jev_alternative_check(jev, instruction, product):
    lines = [
        f"用户需求：{instruction}",
        "候选商品（Agent 实际购买）：",
        f"- 标题：{product.get('title') or ''}",
        f"- 店铺：{product.get('shop_name') or ''}",
        f"- 类目：{product.get('category') or ''}",
        f"- 价格：{product.get('final_price', product.get('pricing', [None])[0])} 元",
        f"- 关键属性：{', '.join(product.get('attribute') or [])}",
    ]
    verdict = jev.decide_satisfaction("\n".join(lines))
    return verdict["choice"]


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
    selected = pick_tasks(tasks, records_by_id)
    OUT.mkdir(parents=True, exist_ok=True)

    teacher = AnthropicTeacherClient(
        api_key=os.environ["TEACHER_API_KEY"],
        base_url=os.environ["TEACHER_BASE_URL"],
        model=os.environ["TEACHER_MODEL"],
    )
    jev = JevDecisionsClient(api_key=os.environ["OPENROUTER_API_KEY"])

    summary = []
    for number, task in enumerate(selected, start=1):
        record = records_by_id[task["record_id"]]
        instruction = record["instructions"][0]["instruction"]
        system = SYSTEM_PROMPT
        if task.get("persona_ref"):
            persona_text = render_persona(pool[task["persona_ref"]]["persona"])
            system = (SYSTEM_PROMPT + "\n\n以下是该用户的画像，属于软偏好参考，"
                      "与当前请求冲突时以请求为准：\n\n" + persona_text)
        env_task = {
            "task_id": index_of[task["record_id"]],
            "prompt": [{"role": "system", "content": system}],
        }
        trajectory = collect_for_task(env_task, client=teacher, base_url="http://127.0.0.1:5700", max_steps=30)
        trajectory["meta"] = {
            "record_id": task["record_id"],
            "difficulty": task["difficulty"],
            "persona_condition": task["persona_condition"],
        }
        with (OUT / f"trajectory_{number:02d}.json").open("w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=1)

        terminal = trajectory.get("terminal_result") or {}
        reward_type = terminal.get("reward_type") or trajectory.get("status")
        steps = len(trajectory.get("steps") or [])
        structural = trajectory.get("status") in STRUCTURAL_OK
        verdict = None
        if terminal.get("reward_type") == "gold_purchase":
            verdict = "gold"
        elif terminal.get("reward_type") in {"valid_alternative_purchase",
                                              "partial_alternative_purchase"}:
            purchase = terminal.get("purchase") or {}
            bought = records_by_id.get(str(purchase.get("asin") or ""))
            if bought:
                try:
                    choice = jev_alternative_check(jev, instruction, bought)
                    verdict = f"jev:{choice}"
                except JevApiError as exc:
                    verdict = f"jev_error:{str(exc)[:40]}"
            else:
                verdict = f"unknown_purchase:{purchase.get('asin')}"
        else:
            verdict = f"env:{reward_type}"
        if not structural:
            verdict = f"ill-formed({trajectory.get('status')}):{verdict}"
        summary.append({
            "record_id": task["record_id"],
            "difficulty": task["difficulty"],
            "condition": task["persona_condition"],
            "status": trajectory.get("status"),
            "steps": steps,
            "reward": trajectory.get("final_reward"),
            "reward_type": reward_type,
            "verdict": verdict,
        })
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    gold = sum(1 for s in summary if s["verdict"] == "gold")
    jev_full = sum(1 for s in summary if s["verdict"] == "jev:fully_satisfies")
    print(f"\n试点完成：gold {gold}，jev fully {jev_full} / {len(summary)}")


if __name__ == "__main__":
    main()
