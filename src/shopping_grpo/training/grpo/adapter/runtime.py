"""每条 veRL trajectory 的轻量运行状态；不保存 ShopSimulator 隐藏 goal。"""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import math
from collections.abc import Mapping


current_environment: ContextVar = ContextVar("shopsimulator_environment", default=None)
current_runtime_state: ContextVar = ContextVar("shopsimulator_runtime_state", default=None)
REWARD_V3_TYPES = {
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "graceful_stop",
    "early_abstain",
    "wrong_purchase",
    "repeat_loop",
    "max_steps",
    "reward_unverifiable",
}


def make_runtime_state(task_id: int, max_steps: int) -> dict:
    """创建只含公共运行诊断的状态，reward 仅在环境正常终局后写入。"""
    return {
        "task_id": int(task_id),
        "max_steps": int(max_steps),
        "steps": [],
        "done": False,
        "terminate": False,
        "termination_reason": None,
        "consecutive_guard_rejections": 0,
        "action_attempt_count": 0,
        "repeat_action_count": 0,
        "recent_action_signatures": [],
        "terminal_result": {},
        "final_reward": 0.0,
        "reward_version": None,
        "reward_type": None,
        "reward_valid": True,
        "reward_unverifiable": False,
        "reward_detail": None,
        "infrastructure_invalid": False,
        "error": None,
        "context_compactions": 0,
        "context_tokens_removed": 0,
        "context_max_input_tokens": 0,
        "observation_projection_count": 0,
        "observation_truncated_count": 0,
        "observation_raw_tokens": 0,
        "observation_visible_tokens": 0,
        "observation_max_raw_tokens": 0,
        "observation_max_visible_tokens": 0,
        "observation_visible_asin_count": 0,
        "observation_visible_button_count": 0,
        "observation_any_truncated": False,
        "latest_observation_truncated": False,
        "observation_footer_failures": 0,
        "guard_rejection_count": 0,
        "guard_rejection_reason_counts": {},
        "guard_rejection_after_truncation_count": 0,
        "action_attempt_after_truncation_count": 0,
    }


def record_observation_projection(state: dict, meta: dict) -> None:
    """Aggregate public projection diagnostics without retaining hidden environment state."""
    raw_tokens = int(meta["raw_tokens"])
    visible_tokens = int(meta["visible_tokens"])
    state["observation_projection_count"] += 1
    state["observation_truncated_count"] += int(bool(meta["truncated"]))
    state["observation_raw_tokens"] += raw_tokens
    state["observation_visible_tokens"] += visible_tokens
    state["observation_max_raw_tokens"] = max(state["observation_max_raw_tokens"], raw_tokens)
    state["observation_max_visible_tokens"] = max(
        state["observation_max_visible_tokens"], visible_tokens
    )
    state["observation_visible_asin_count"] += int(meta["visible_asin_count"])
    state["observation_visible_button_count"] += int(meta["visible_button_count"])
    state["observation_any_truncated"] = (
        state["observation_any_truncated"] or bool(meta["truncated"])
    )
    state["latest_observation_truncated"] = bool(meta["truncated"])
    state["observation_footer_failures"] += int(
        not bool(meta["critical_footer_preserved"])
    )


def record_action_attempt(state: dict, tool_name: str, parameters: dict, observation: str) -> None:
    """记录环境动作尝试，并统计最近三次中的重复签名。"""
    if tool_name == "think":
        return
    canonical_parameters = json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    observation_fingerprint = hashlib.sha256(str(observation).encode("utf-8")).hexdigest()
    signature = (str(tool_name), canonical_parameters, observation_fingerprint)
    recent = state["recent_action_signatures"]
    state["action_attempt_count"] += 1
    if signature in recent:
        state["repeat_action_count"] += 1
    recent.append(signature)
    del recent[:-3]


def validate_reward(raw_detail: object) -> dict:
    """Validate and minimize public Environment v2.1 / Reward v3 diagnostics."""
    if not isinstance(raw_detail, Mapping):
        raise ValueError("reward_detail must be an object")
    if raw_detail.get("reward_version") != "shopsimulator-reward-v3":
        raise ValueError("reward_detail has an unsupported reward_version")
    reward_type = str(raw_detail.get("reward_type", ""))
    if reward_type not in REWARD_V3_TYPES:
        raise ValueError(f"unknown Reward v3 reward_type: {reward_type!r}")
    if raw_detail.get("termination_reason") != reward_type:
        raise ValueError("termination_reason must equal reward_type")
    reward_valid = raw_detail.get("reward_valid")
    if not isinstance(reward_valid, bool):
        raise ValueError("reward_valid must be boolean")
    if (reward_type == "reward_unverifiable") != (not reward_valid):
        raise ValueError("only reward_unverifiable may set reward_valid=false")
    try:
        terminal_utility = float(raw_detail.get("terminal_utility"))
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal_utility must be numeric") from exc
    if not math.isfinite(terminal_utility):
        raise ValueError("terminal_utility must be finite")
    purchase_success = raw_detail.get("purchase_success")
    sampling_invalid = raw_detail.get("sampling_invalid")
    if not isinstance(purchase_success, bool):
        raise ValueError("purchase_success must be boolean")
    if not isinstance(sampling_invalid, bool):
        raise ValueError("sampling_invalid must be boolean")
    if sampling_invalid != (not reward_valid):
        raise ValueError("sampling_invalid must equal not reward_valid")
    hard_gates = raw_detail.get("hard_gates")
    if not isinstance(hard_gates, Mapping):
        raise ValueError("hard_gates must be an object")
    public_gates = {}
    for name, raw_gate in hard_gates.items():
        if not isinstance(raw_gate, Mapping):
            raise ValueError(f"hard gate {name!r} must be an object")
        status = raw_gate.get("status")
        if status not in {"pass", "fail", "unverifiable"}:
            raise ValueError(f"hard gate {name!r} has invalid status")
        if raw_gate.get("passed") != (status == "pass"):
            raise ValueError(f"hard gate {name!r} has inconsistent passed")
        if raw_gate.get("verifiable") != (status != "unverifiable"):
            raise ValueError(f"hard gate {name!r} has inconsistent verifiable")
        public_gates[str(name)] = {
            "status": status,
            "passed": raw_gate["passed"],
            "verifiable": raw_gate["verifiable"],
            "comparator": str(raw_gate.get("comparator") or ""),
            "source_field": str(raw_gate.get("source_field") or ""),
        }
    try:
        weighted_score = float(raw_detail.get("weighted_score", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("weighted_score must be numeric") from exc
    if not math.isfinite(weighted_score) or not 0.0 <= weighted_score <= 1.0:
        raise ValueError("weighted_score must be finite and in [0, 1]")
    try:
        evidence_coverage = float(raw_detail.get("evidence_coverage", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence_coverage must be numeric") from exc
    if (
        not math.isfinite(evidence_coverage)
        or not 0.0 <= evidence_coverage <= 1.0
    ):
        raise ValueError("evidence_coverage must be finite and in [0, 1]")
    raw_dimension_scores = raw_detail.get("dimension_scores") or {}
    if not isinstance(raw_dimension_scores, Mapping):
        raise ValueError("dimension_scores must be an object")
    dimension_scores = {}
    for name in ("brand", "model", "core_functions", "key_options"):
        try:
            score = float(raw_dimension_scores.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dimension score {name} must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"dimension score {name} must be finite and in [0, 1]"
            )
        dimension_scores[name] = score
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "termination_reason": reward_type,
        "target_asin_match": bool(raw_detail.get("target_asin_match")),
        "hard_gates": public_gates,
        "weighted_score": weighted_score,
        "evidence_coverage": evidence_coverage,
        "dimension_scores": dimension_scores,
        "terminal_utility": terminal_utility,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
    }


def _normal_terminal(state: dict) -> bool:
    terminal = state.get("terminal_result") or {}
    return (
        state.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
    )


def reward_breakdown(state: dict) -> dict[str, float | bool]:
    """计算约束感知终局奖励；基础设施无效轨迹只返回诊断，不制造学习信号。"""
    invalid = bool(state.get("infrastructure_invalid"))
    normal_terminal = _normal_terminal(state)
    native = float(state.get("final_reward", 0.0)) if normal_terminal else 0.0
    if not math.isfinite(native):
        invalid = True
        native = 0.0

    if state.get("reward_version") == "shopsimulator-reward-v3":
        detail = state.get("reward_detail") or {}
        reward_valid = bool(state.get("reward_valid", True))
        invalid_reward = not reward_valid
        gates = detail.get("hard_gates") or {}
        dimension_scores = detail.get("dimension_scores") or {}
        component = lambda name: float(bool(gates.get(name, {}).get("passed")))
        full = float(state.get("reward_type") == "gold_purchase")
        strict = full
        purchase_success = bool(
            state.get("reward_type")
            in {"gold_purchase", "valid_alternative_purchase"}
        )
        terminal_utility = (
            native if normal_terminal and not invalid and not invalid_reward else 0.0
        )
        semantic = float(purchase_success)
        return {
            "r_type": component("category"),
            "r_att": float(detail.get("weighted_score", 0.0)),
            "r_option": float(dimension_scores.get("key_options", 0.0)),
            "r_price": component("budget"),
            "match_score": float(detail.get("weighted_score", 0.0)),
            "evidence_coverage": float(detail.get("evidence_coverage", 0.0)),
            "brand_score": float(dimension_scores.get("brand", 0.0)),
            "model_score": float(dimension_scores.get("model", 0.0)),
            "core_function_score": float(
                dimension_scores.get("core_functions", 0.0)
            ),
            "option_score": float(
                dimension_scores.get("key_options", 0.0)
            ),
            "full": full,
            "strict": strict,
            "native": native,
            "semantic": semantic,
            "efficiency": 0.0,
            "penalty_overlong": 0.0,
            "penalty_unfinished": 0.0,
            "penalty_repeat": 0.0,
            "repeat_action_rate": (
                int(state.get("repeat_action_count", 0))
                / max(int(state.get("action_attempt_count", 0)), 1)
            ),
            "total": terminal_utility,
            "terminal_utility": terminal_utility,
            "purchase_success": float(purchase_success),
            "sampling_invalid": bool(invalid or invalid_reward),
            "infrastructure_invalid": invalid,
            "reward_unverifiable": invalid_reward,
        }

    action_attempts = max(int(state.get("action_attempt_count", 0)), 1)
    repeat_action_rate = int(state.get("repeat_action_count", 0)) / action_attempts
    return {
        "r_type": 0.0,
        "r_att": 0.0,
        "r_option": 0.0,
        "r_price": 0.0,
        "match_score": 0.0,
        "evidence_coverage": 0.0,
        "brand_score": 0.0,
        "model_score": 0.0,
        "core_function_score": 0.0,
        "option_score": 0.0,
        "full": 0.0,
        "strict": 0.0,
        "native": native,
        "semantic": 0.0,
        "efficiency": 0.0,
        "penalty_overlong": 0.0,
        "penalty_unfinished": 0.0,
        "penalty_repeat": 0.0,
        "repeat_action_rate": repeat_action_rate,
        "total": 0.0,
        "terminal_utility": 0.0,
        "purchase_success": 0.0,
        "sampling_invalid": True,
        "infrastructure_invalid": True,
        "reward_unverifiable": True,
    }


def apply_reward_length_shaping(
    reward: Mapping,
    state: Mapping,
    *,
    enabled: bool,
    soft_threshold: int = 20,
    penalty_per_step: float = 0.01,
    max_penalty: float = 0.15,
) -> dict:
    """Apply optional soft length cost and invalidate hard max-step trajectories."""
    shaped = dict(reward)
    if not enabled:
        return shaped
    threshold = int(soft_threshold)
    per_step = float(penalty_per_step)
    cap = float(max_penalty)
    if threshold < 1 or per_step < 0 or cap < 0:
        raise ValueError("length shaping threshold must be positive and penalties non-negative")

    steps = len(state.get("steps") or [])
    penalty = min(max(steps - threshold, 0) * per_step, cap)
    shaped["penalty_overlong"] = float(shaped.get("penalty_overlong", 0.0)) + penalty
    shaped["total"] = float(shaped["total"]) - penalty
    shaped["terminal_utility"] = float(shaped["terminal_utility"]) - penalty
    shaped["overlong"] = bool(
        state.get("termination_reason") == "max_steps"
        or steps >= int(state.get("max_steps", steps + 1))
    )
    if shaped["overlong"]:
        shaped["sampling_invalid"] = True
    return shaped


def terminal_reward(state: dict, mode: str = "native") -> float:
    """按实验模式返回原生或约束感知奖励。"""
    if mode == "constraint_aware":
        return float(reward_breakdown(state)["total"])
    if mode != "native":
        raise ValueError(f"unknown shopping reward mode: {mode!r}")
    if state.get("infrastructure_invalid") or state.get("error") or not _normal_terminal(state):
        return 0.0
    return float(state.get("final_reward", 0.0))


def task_id_from_kwargs(kwargs: dict) -> int:
    """从 veRL parquet 的 extra_info 读取当前任务，缺失时立即失败。"""
    extra_info = kwargs.get("extra_info")
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, dict) or "task_id" not in extra_info:
        raise ValueError("veRL sample extra_info is missing task_id")
    return int(extra_info["task_id"])
