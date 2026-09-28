import pytest

from shopping_grpo.evaluation.jev import build_calibration_report, score_trajectories


def _trajectory(task_id, reward_type, reward_valid=True):
    return {
        "task_id": task_id,
        "trajectory_id": f"trajectory-{task_id}",
        "status": "done",
        "done": True,
        "steps": [{"tool_name": "search_products"}],
        "terminal_result": {
            "done": True,
            "over": True,
            "purchase": {"asin": f"A{task_id}"},
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": reward_type,
                "reward_valid": reward_valid,
                "purchase_success": True,
                "termination_reason": reward_type,
            },
        },
    }


def test_jev_scoring_and_calibration_require_valid_judgments():
    trajectories = [
        _trajectory(1, "gold_purchase"),
        _trajectory(2, "partial_alternative_purchase"),
        _trajectory(3, "reward_unverifiable", False),
    ]
    metadata = {
        task_id: {
            "difficulty": "easy",
            "persona_condition": "no_profile",
            "domain": "测试领域",
        }
        for task_id in (1, 2, 3)
    }
    rows, summary = score_trajectories(
        trajectories,
        metadata,
        {"alternative": {"ok": True, "choice": "fully_satisfies"}},
        {2: {"request_hash": "alternative"}},
    )

    assert summary["scorable_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["completion_rate"] == 1.0
    assert rows[1]["outcome"] == "jev_fully_satisfies"
    assert rows[2]["invalid_reason"] == "reward_unverifiable"

    cases = [
        {
            "calibration_id": "case-1",
            "calibration_source": "grey",
            "task_id": 2,
            "request_hash": "alternative",
            "state": "case state",
        }
    ]
    _, review, report = build_calibration_report(
        cases,
        {
            "alternative": {
                "ok": True,
                "choice": "fully_satisfies",
                "confidence": 0.7,
            }
        },
        {"case-1": {"ok": True, "choice": "partially_satisfies"}},
        review_target=1,
    )

    assert len(review) == 1
    assert report["human_completed"] == 0
    assert report["status"] == "pending_human_adjudication"

    with pytest.raises(ValueError, match="invalid human choice"):
        build_calibration_report(
            cases,
            {"alternative": {"ok": True, "choice": "fully_satisfies"}},
            {"case-1": {"ok": True, "choice": "partially_satisfies"}},
            review_target=1,
            human_by_id={"case-1": {"human_choice": "invalid"}},
        )
