#!/usr/bin/env python3
"""Fail fast on the veRL APIs required by the Shopping GRPO adapter."""

from __future__ import annotations

import os
from pathlib import Path


def main():
    required_paths = {
        "GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE"),
        "GRPO_VAL_FILE": os.environ.get("GRPO_VAL_FILE"),
    }
    missing = [name for name, value in required_paths.items() if not value or not Path(value).is_file()]
    if missing:
        raise SystemExit("missing GRPO parquet file(s): " + ", ".join(missing))

    try:
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
        from shopping_grpo.verl_adapter.agent_loop import ShoppingToolAgentLoop
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL install: async ToolAgentLoop APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_interacting_state")
        or not callable(initialize_interactions_from_config)
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("shopping qwen3_coder parser was not registered")
    print("GRPO runtime preflight passed: datasets, veRL AgentLoop, parser")


if __name__ == "__main__":
    main()
