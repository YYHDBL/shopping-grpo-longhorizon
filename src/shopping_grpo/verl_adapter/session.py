"""一条 veRL trajectory 对应一个 ShopSimulator 环境租约。"""

from __future__ import annotations

import asyncio

from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.verl_adapter.runtime import current_environment, current_runtime_state, make_runtime_state


class ShopSimulatorSession:
    """负责 reset、绑定 coroutine-local 状态，并保证 release。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5700",
        timeout: int = 60,
        max_steps: int = 35,
        env_factory=None,
    ):
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.env_factory = env_factory or ShopAgentEnv
        self.env = None
        self.state = None
        self._environment_token = None
        self._state_token = None

    async def start(self, task_id: int) -> dict:
        if self.env is not None:
            raise RuntimeError("ShopSimulator session has already started")
        self.env = self.env_factory(base_url=self.base_url, timeout=self.timeout)
        try:
            initial = await asyncio.to_thread(self.env.reset, int(task_id))
        except Exception:
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise

        self.state = make_runtime_state(task_id=task_id, max_steps=self.max_steps)
        self.state["latest_observation"] = str(
            initial.get("instruction", initial.get("observation", "")) if isinstance(initial, dict) else initial
        )
        self._environment_token = current_environment.set(self.env)
        self._state_token = current_runtime_state.set(self.state)
        return self.state

    async def close(self) -> None:
        if self.env is None:
            return
        try:
            await asyncio.to_thread(self.env.release)
        except Exception as exc:
            if self.state is not None:
                self.state["error"] = f"release_error:{exc.__class__.__name__}:{exc}"
            raise
        finally:
            if self._state_token is not None:
                current_runtime_state.reset(self._state_token)
            if self._environment_token is not None:
                current_environment.reset(self._environment_token)
            self.env = None
            self._state_token = None
            self._environment_token = None
