"""Deterministic trajectory acceptance and SFT message construction."""

import json
from collections import Counter
from pathlib import Path

from shopping_grpo.action_validation import RUNTIME_GUARD_FIELD, action_reject_reason
from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


REWARD_KEYS = ("r_type", "r_att", "r_option", "r_price")
ALLOWED_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
ALLOWED_TOOL_CALL_KEYS = {"id", "type", "function"}
ALLOWED_FUNCTION_KEYS = {"name", "arguments"}


def acceptance_reasons(trajectory):
    reasons = []
    steps = trajectory.get("steps") or []
    terminal = trajectory.get("terminal_result") or {}
    reward_detail = terminal.get("reward_detail") or {}

    if trajectory.get("error"):
        reasons.append("has_error")
    if trajectory.get("status") != "done":
        reasons.append("status_not_done")
    if not trajectory.get("done"):
        reasons.append("trajectory_not_done")
    if not terminal.get("done") or not terminal.get("over"):
        reasons.append("environment_not_done")
    if not any(step.get("tool_name") == "buy_now" or step.get("env_action") == "click[Buy Now]" for step in steps):
        reasons.append("missing_buy")
    if not isinstance(reward_detail, dict) or any(key not in reward_detail for key in REWARD_KEYS):
        reasons.append("reward_detail_incomplete")
    else:
        for key in REWARD_KEYS:
            if reward_detail.get(key) != 1 and reward_detail.get(key) is not True:
                reasons.append(f"reward_detail.{key}_not_1")

    for index, message in enumerate(trajectory.get("messages") or []):
        if message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1:
            reasons.append(f"message_{index}.multiple_tool_calls")

    option_selected = False
    for index, step in enumerate(steps):
        reason = _tool_step_reject_reason(trajectory, step)
        if reason:
            reasons.append(f"step_{index}.{reason}")
        tool_name = step.get("tool_name")
        if option_selected and tool_name not in {"select_option", "buy_now", "think"}:
            reasons.append(f"step_{index}.action_after_option_selection")
        if tool_name == "select_option":
            option_selected = True

    return len(reasons) == 0, reasons


def build_sft_row(trajectory, retain_reasoning=False):
    """构造 SFT 行；默认只学习动作，不模仿 Teacher 的长思考。"""
    terminal_tool_call_id = _terminal_tool_call_id(trajectory)
    blocked_call_ids = {
        (blocked.get("tool_call") or {}).get("id")
        for blocked in trajectory.get("blocked_tool_calls") or []
    }
    blocked_call_ids.discard(None)
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "task_id": trajectory.get("task_id"),
        "messages": _training_messages(
            trajectory.get("messages", []),
            blocked_call_ids,
            terminal_tool_call_id,
            retain_reasoning=retain_reasoning,
        ),
        "tools": SHOP_TOOL_SCHEMAS,
    }


def _training_messages(messages, blocked_call_ids, terminal_tool_call_id, retain_reasoning=False):
    """移除守卫对话；默认不把 Teacher reasoning 写进训练目标。"""
    clean_messages = []
    follows_runtime_guard = False
    for message in messages:
        if _is_blocked_training_message(message, blocked_call_ids):
            follows_runtime_guard = message.get(RUNTIME_GUARD_FIELD) is True
            continue
        clean_messages.append(
            _sanitize_message(
                message,
                terminal_tool_call_id,
                retain_reasoning=retain_reasoning and not follows_runtime_guard,
            )
        )
        follows_runtime_guard = False
    return clean_messages


def _is_blocked_training_message(message, blocked_call_ids):
    if message.get(RUNTIME_GUARD_FIELD) is True:
        return True
    if message.get("role") == "tool" and message.get("tool_call_id") in blocked_call_ids:
        return True
    if message.get("role") == "assistant":
        return any(call.get("id") in blocked_call_ids for call in message.get("tool_calls") or [])
    return False


def process_raw_trajectories(
    raw_path,
    accepted_path,
    rejected_path,
    stats_path,
    sft_path,
    retain_reasoning=False,
):
    accepted_rows = []
    rejected_rows = []
    sft_rows = []
    reason_counts = Counter()

    for trajectory in _read_jsonl(raw_path):
        accepted, reasons = acceptance_reasons(trajectory)
        if accepted:
            accepted_rows.append(trajectory)
            sft_rows.append(build_sft_row(trajectory, retain_reasoning=retain_reasoning))
        else:
            rejected = {
                "trajectory_id": trajectory.get("trajectory_id"),
                "task_id": trajectory.get("task_id"),
                "status": trajectory.get("status"),
                "reject_reasons": reasons,
            }
            rejected_rows.append(rejected)
            reason_counts.update(reasons)

    _write_jsonl(accepted_path, accepted_rows)
    _write_jsonl(rejected_path, rejected_rows)
    _write_jsonl(sft_path, sft_rows)
    summary = {
        "total": len(accepted_rows) + len(rejected_rows),
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "reject_reasons": dict(sorted(reason_counts.items())),
        "retain_teacher_reasoning": bool(retain_reasoning),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _tool_step_reject_reason(trajectory, step):
    """校验工具参数与动作，并确保点击来自紧邻的环境 observation。"""
    tool_call = step.get("tool_call") or {}
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        return "missing_tool_name"
    try:
        arguments = json.loads(raw_arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return "invalid_arguments_json"
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if name != step.get("tool_name"):
        return "tool_name_mismatch"
    try:
        expected_action = tool_call_to_action(name, arguments)
    except Exception:
        return "unknown_or_invalid_tool"
    if expected_action != step.get("env_action"):
        return "env_action_mismatch"
    return action_reject_reason(
        name,
        arguments,
        _previous_observation(trajectory, tool_call.get("id")),
    )


def _previous_observation(trajectory, tool_call_id):
    """返回此工具调用前最近一次 tool message，缺失时返回空字符串。"""
    if not tool_call_id:
        return ""
    messages = trajectory.get("messages") or []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not any(call.get("id") == tool_call_id for call in calls):
            continue
        for previous in reversed(messages[:index]):
            if previous.get(RUNTIME_GUARD_FIELD) is True:
                continue
            if previous.get("role") == "tool":
                content = previous.get("content")
                return content if isinstance(content, str) else ""
        return ""
    return ""


def _terminal_tool_call_id(trajectory):
    """返回环境结束的工具调用 id，用于移除终局隐藏反馈。"""
    if not trajectory.get("done"):
        return None
    terminal_steps = [step for step in trajectory.get("steps", []) if step.get("done")]
    if not terminal_steps:
        return None
    return (terminal_steps[-1].get("tool_call") or {}).get("id")


def _sanitize_message(message, terminal_tool_call_id=None, retain_reasoning=True):
    clean = {key: message[key] for key in ALLOWED_MESSAGE_KEYS if key in message}
    reasoning = message.get("reasoning_content")
    if (
        retain_reasoning
        and clean.get("role") == "assistant"
        and isinstance(reasoning, str)
        and reasoning.strip()
    ):
        # 用标准 content 保存教师推理，避免 chat template 忽略 provider 专有字段。
        thought = f"<think>{reasoning.strip()}</think>"
        content = clean.get("content")
        clean["content"] = thought if not content else f"{thought}\n{content}"
    if clean.get("role") == "tool" and clean.get("tool_call_id") == terminal_tool_call_id:
        clean["content"] = "购买已完成。"
    if "tool_calls" in clean:
        clean["tool_calls"] = [_sanitize_tool_call(call) for call in clean["tool_calls"]]
    return clean


def _sanitize_tool_call(tool_call):
    clean = {key: tool_call[key] for key in ALLOWED_TOOL_CALL_KEYS if key in tool_call}
    if "function" in clean:
        clean["function"] = {
            key: clean["function"][key]
            for key in ALLOWED_FUNCTION_KEYS
            if key in clean["function"]
        }
    return clean


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
