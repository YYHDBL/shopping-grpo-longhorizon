"""veRL 原生工具适配：复用本项目唯一的 ShopSimulator tool schema 与动作守卫。"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from shopping_grpo.action_validation import action_reject_reason
from shopping_grpo.shop_tools import tool_call_to_action
from shopping_grpo.verl_adapter.runtime import current_environment, current_runtime_state

try:  # 本地单测不安装 veRL；部署时由 veRL 注入真实类型。
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
    from verl.utils.rollout_trace import rollout_trace_op
except ImportError:  # pragma: no cover - 仅轻量开发环境使用
    class ToolResponse:
        def __init__(self, text=None, image=None, video=None):
            self.text, self.image, self.video = text, image, video

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config, self.tool_schema = config, tool_schema
            function = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else tool_schema.function
            self.name = function.get("name") if isinstance(function, dict) else function.name

    def rollout_trace_op(function):
        return function


class ShopSimulatorTool(BaseTool):
    """共享工具定义；当前 coroutine 的 env/state 由 AgentLoop 绑定。"""

    async def create(self, instance_id=None, **kwargs):
        del kwargs
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        del instance_id, kwargs
        env = current_environment.get()
        state = current_runtime_state.get()
        if env is None or state is None:
            raise RuntimeError("ShopSimulator tool executed without a trajectory-local interaction state")
        if state["done"] or state["terminate"]:
            return ToolResponse(text="Error: environment is already terminal; do not call another tool."), 0.0, {}
        if len(state["steps"]) >= state["max_steps"]:
            _terminate(state, "max_steps")
            return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, {"reason": "max_steps"}
        parameters = parameters if isinstance(parameters, dict) else {}
        if self.name == "think":
            step = _append_step(state, self.name, parameters)
            if len(state["steps"]) >= state["max_steps"]:
                _terminate(state, "max_steps")
                return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, step
            return ToolResponse(text="Reasoning recorded. Continue with one environment tool call."), 0.0, step
        observation = state.get("latest_observation", "")
        reason = action_reject_reason(self.name, parameters, observation)
        if reason:
            state["consecutive_guard_rejections"] += 1
            if state["consecutive_guard_rejections"] >= 3:
                _terminate(state, "too_many_guard_rejections")
                return ToolResponse(text="Error: maximum consecutive action guard rejections reached."), 0.0, {
                    "reason": reason
                }
            return ToolResponse(text=f"Error: action guard rejected this call ({reason}); read the latest observation."), 0.0, {"reason": reason}
        try:
            action = tool_call_to_action(self.name, parameters)
            result = await asyncio.to_thread(env.step, action)
            observation = str(result.get("instruction", result.get("observation", "")))
            step = _append_step(
                state,
                self.name,
                parameters,
                done=bool(result.get("done", False)),
                reward=float(result.get("reward", 0.0)),
            )
        except Exception as exc:
            _terminate(state, f"tool_error:{exc.__class__.__name__}:{exc}")
            return ToolResponse(text=f"Error: ShopSimulator tool execution failed: {exc}"), 0.0, {"error": state["error"]}
        state["consecutive_guard_rejections"] = 0
        if step["done"]:
            state["done"] = True
            state["terminate"] = True
            state["termination_reason"] = "environment_done"
            state["terminal_result"] = {
                "done": True,
                "over": result.get("over") is True,
            }
            state["final_reward"] = step["reward"]
            return ToolResponse(text="Environment terminated."), 0.0, step
        state["latest_observation"] = observation
        if len(state["steps"]) >= state["max_steps"]:
            _terminate(state, "max_steps")
            return ToolResponse(text="Error: maximum executed tool steps reached."), 0.0, step
        return ToolResponse(text=observation), 0.0, step

    async def release(self, instance_id, **kwargs):
        # veRL 0.8 会在每次 tool call 后执行 release；真正的环境租约由 AgentLoop 释放。
        del instance_id, kwargs


def _append_step(state, tool, parameters, done=False, reward=0.0):
    step = {
        "index": len(state["steps"]),
        "tool": tool,
        "parameters": parameters,
        "done": bool(done),
        "reward": float(reward),
    }
    state["steps"].append(step)
    return step


def _terminate(state, reason):
    state["terminate"] = True
    state["termination_reason"] = reason
    state["error"] = reason
