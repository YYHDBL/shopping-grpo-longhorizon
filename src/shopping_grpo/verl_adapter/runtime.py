"""每条 veRL trajectory 的轻量运行状态；不保存 ShopSimulator 隐藏 goal。"""

from __future__ import annotations

from contextvars import ContextVar


current_environment: ContextVar = ContextVar("shopsimulator_environment", default=None)
current_runtime_state: ContextVar = ContextVar("shopsimulator_runtime_state", default=None)


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
        "terminal_result": {},
        "final_reward": 0.0,
        "error": None,
    }


def terminal_reward(state: dict) -> float:
    """Vanilla GRPO 的唯一奖励：ShopSimulator 原生正常终局 reward。"""
    terminal = state.get("terminal_result") or {}
    if (
        state.get("error")
        or not state.get("done")
        or terminal.get("done") is not True
        or terminal.get("over") is not True
    ):
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
