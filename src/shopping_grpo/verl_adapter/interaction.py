"""把每条 veRL trajectory 绑定到一份独占的 ShopSimulator HTTP 环境租约。"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.verl_adapter.runtime import (
    current_environment,
    current_interaction_instance_id,
    current_runtime_state,
    make_runtime_state,
    terminal_reward,
)

try:  # 本地单测无需安装重型 veRL；服务器会使用真实基类。
    from verl.interactions.base import BaseInteraction
except ImportError:  # pragma: no cover - 仅轻量开发环境使用
    class BaseInteraction:
        def __init__(self, config):
            self.config = config


class ShopSimulatorInteraction(BaseInteraction):
    """veRL lifecycle adapter：reset 在开始、release 永远在 finalize。"""

    def __init__(self, config, env_factory=None):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://127.0.0.1:5700")
        self.timeout = int(config.get("timeout", 60))
        self.max_steps = int(config.get("max_steps", 35))
        self.env_factory = env_factory or ShopAgentEnv
        self._instances = {}

    async def start_interaction(self, instance_id=None, task_id=0, **kwargs):
        del kwargs
        instance_id = instance_id or str(uuid4())
        env = self.env_factory(base_url=self.base_url, timeout=self.timeout)
        try:
            initial = await asyncio.to_thread(env.reset, int(task_id))
        except Exception:
            try:
                await asyncio.to_thread(env.release)
            except Exception:
                pass
            raise
        state = make_runtime_state(task_id=task_id, max_steps=self.max_steps)
        # 这是用户可见的当前 observation，不是 goal、标准答案或 reward_detail。
        state["latest_observation"] = initial.get("instruction", initial.get("observation", ""))
        self._instances[instance_id] = {"env": env, "state": state}
        current_environment.set(env)
        current_runtime_state.set(state)
        current_interaction_instance_id.set(instance_id)
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        """工具终局或无工具 assistant 输出时结束，并把评分留给 calculate_score。"""
        del messages, kwargs
        entry = self._instances.get(instance_id)
        if entry is None:
            raise RuntimeError(f"missing ShopSimulator interaction instance: {instance_id}")
        current_environment.set(entry["env"])
        current_runtime_state.set(entry["state"])
        state = entry["state"]
        if not state["done"]:
            state["error"] = state["error"] or "assistant_finished_without_environment_done"
            state["terminate"] = True
            state["termination_reason"] = state["error"]
        return True, "", 0.0, {
            "task_id": state["task_id"],
            "steps": len(state["steps"]),
            "terminal_reward": terminal_reward(state),
            "error": state["error"],
        }

    async def calculate_score(self, instance_id, **kwargs):
        del kwargs
        entry = self._instances.get(instance_id)
        return terminal_reward(entry["state"]) if entry else 0.0

    async def finalize_interaction(self, instance_id, **kwargs):
        del kwargs
        entry = self._instances.get(instance_id)
        if entry is None:
            return
        try:
            await asyncio.to_thread(entry["env"].release)
        except Exception as exc:  # 租约可能仍占用；保留实例并让训练 fail fast。
            entry["state"]["error"] = f"release_error:{exc.__class__.__name__}:{exc}"
            raise
        self._instances.pop(instance_id, None)
        if current_interaction_instance_id.get() == instance_id:
            current_interaction_instance_id.set(None)
            current_environment.set(None)
            current_runtime_state.set(None)

    async def finalize_current_interaction(self):
        """供项目 AgentLoop 的 finally 调用，覆盖 veRL 未暴露 request_id 的异常路径。"""
        instance_id = current_interaction_instance_id.get()
        if instance_id is not None:
            await self.finalize_interaction(instance_id)
