import json

from scripts.evaluate_model import summarize


def test_summary_uses_frozen_tasks_for_overall_and_bucket_denominators(tmp_path):
    trajectory = {
        "task_id": 10,
        "status": "done",
        "done": True,
        "final_reward": 1.0,
        "steps": [{"tool_name": "buy_now", "done": True, "reward": 1.0}],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        },
    }
    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    tasks = [
        {"task_id": 10, "difficulty": "easy", "persona_condition": "aligned", "domain": "服饰"},
        {"task_id": 11, "difficulty": "hard", "persona_condition": "no_profile", "domain": "家居"},
    ]

    summary = summarize(path, tasks)

    assert summary["expected_task_count"] == 2
    assert summary["strict_gold_success_rate"] == 0.5
    assert summary["by_difficulty"]["easy"]["strict_gold_success_rate"] == 1.0
    assert summary["by_difficulty"]["hard"]["strict_gold_success_rate"] == 0.0
    assert summary["by_persona_condition"]["aligned"]["strict_gold_success_count"] == 1
    assert summary["by_domain"]["家居"]["task_count"] == 1
