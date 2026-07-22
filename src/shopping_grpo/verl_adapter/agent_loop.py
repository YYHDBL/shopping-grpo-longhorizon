"""Shopping-specific lifecycle fixes layered on veRL's standard ToolAgentLoop."""

from __future__ import annotations

# Importing registers qwen3_coder before ToolAgentLoop.init_class resolves the parser.
from shopping_grpo.verl_adapter import qwen3_coder_parser as _qwen3_coder_parser  # noqa: F401
from shopping_grpo.verl_adapter.runtime import current_runtime_state
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


class ShoppingToolAgentLoop(ToolAgentLoop):
    """Vanilla ToolAgentLoop with deterministic ShopSimulator termination and release."""

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
        try:
            return await super().run(sampling_params, **kwargs)
        finally:
            for interaction in getattr(self, "interaction_map", {}).values():
                finalize = getattr(interaction, "finalize_current_interaction", None)
                if finalize is not None:
                    await finalize()
