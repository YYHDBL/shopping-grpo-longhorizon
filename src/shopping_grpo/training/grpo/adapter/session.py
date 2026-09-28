"""把一条 veRL trajectory 绑定到一个 ShopSimulator 环境租约。

session 负责异步训练框架与同步 HTTP 客户端之间的边界：在线程中执行网络调用，
在 coroutine-local context 中暴露当前环境和运行状态，并在任何退出路径释放租约。
"""

from __future__ import annotations

import asyncio

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.training.grpo.adapter.runtime import current_environment, current_runtime_state, make_runtime_state


class ShopSimulatorSession:
    """负责 reset、绑定 coroutine-local 状态，并保证 release。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5700",
        timeout: int = 60,
        max_steps: int = 35,
        required_environment_version: str | None = None,
        env_factory=None,
    ):
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.required_environment_version = required_environment_version
        self.env_factory = env_factory or ShopAgentEnv
        self.env = None
        self.state = None
        self._environment_token = None
        self._state_token = None

    async def start(self, task_id: int) -> dict:
        """启动一条 trajectory，并把首个 observation 放进运行状态。"""
        if self.env is not None:
            raise RuntimeError("ShopSimulator session has already started")
        self.env = self.env_factory(base_url=self.base_url, timeout=self.timeout)
        try:
            # ShopAgentEnv 使用阻塞 urllib；放到线程中不会阻塞 veRL 的事件循环。
            initial = await asyncio.to_thread(self.env.reset, int(task_id))
        except Exception:
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise

        self.state = make_runtime_state(task_id=task_id, max_steps=self.max_steps)
        actual_version = (
            initial.get("environment_version") if isinstance(initial, dict) else None
        )
        if (
            self.required_environment_version is not None
            and actual_version != self.required_environment_version
        ):
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise RuntimeError(
                "ShopSimulator environment version mismatch: "
                f"expected {self.required_environment_version!r}, got {actual_version!r}"
            )
        if isinstance(initial, dict) and initial.get("observation_state") is not None:
            self.state["latest_observation"] = render_structured_observation(
                initial["observation_state"]
            )
            self.state["environment_version"] = actual_version
        else:
            self.state["latest_observation"] = str(
                initial.get("instruction", initial.get("observation", ""))
                if isinstance(initial, dict)
                else initial
            )
        # ContextVar 让并发 trajectory 互不串状态，比共享全局 current_env 安全。
        self._environment_token = current_environment.set(self.env)
        self._state_token = current_runtime_state.set(self.state)
        return self.state

    async def close(self) -> None:
        """释放环境并恢复 ContextVar；释放失败仍会清理本地绑定。"""
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
