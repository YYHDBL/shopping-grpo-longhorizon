"""veRL 0.8 ToolAgentLoop 的 ShopSimulator 轨迹生命周期适配。"""

from __future__ import annotations

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from shopping_grpo.verl_adapter.runtime import (
    current_runtime_state,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.verl_adapter.session import ShopSimulatorSession


class ShoppingToolAgentLoop(ToolAgentLoop):
    """Vanilla ToolAgentLoop with deterministic ShopSimulator termination and release."""

    def __init__(
        self,
        *args,
        base_url="http://127.0.0.1:5700",
        timeout=60,
        max_steps=35,
        env_factory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.env_factory = env_factory

    async def _handle_processing_tools_state(self, agent_data):
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and len(agent_data.tool_calls) > 1:
            runtime_state["terminate"] = True
            runtime_state["termination_reason"] = "parallel_tool_calls"
            runtime_state["error"] = "parallel_tool_calls"
            return AgentState.TERMINATED
        next_state = await super()._handle_processing_tools_state(agent_data)
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and runtime_state.get("terminate"):
            return AgentState.TERMINATED
        return next_state

    async def run(self, sampling_params, **kwargs):
        task_id = task_id_from_kwargs(kwargs)
        session = ShopSimulatorSession(
            base_url=self.base_url,
            timeout=self.timeout,
            max_steps=self.max_steps,
            env_factory=self.env_factory,
        )
        state = await session.start(task_id)
        try:
            output = await super().run(sampling_params, **kwargs)
            if not state["done"] and not state["error"]:
                state["error"] = "assistant_finished_without_environment_done"
                state["termination_reason"] = state["error"]
                state["terminate"] = True
            output.reward_score = terminal_reward(state)
            output.extra_fields["shopping"] = {
                "task_id": task_id,
                "steps": len(state["steps"]),
                "termination_reason": state["termination_reason"],
                "error": state["error"],
            }
            return output
        finally:
            await session.close()
