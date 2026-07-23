#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "transformers": "5.11.0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
}


def main():
    required_paths = {
        "GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE"),
        "GRPO_VAL_FILE": os.environ.get("GRPO_VAL_FILE"),
    }
    missing = [name for name, value in required_paths.items() if not value or not Path(value).is_file()]
    if missing:
        raise SystemExit("missing GRPO parquet file(s): " + ", ".join(missing))

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"incompatible Python: expected 3.12, got {sys.version.split()[0]}")

    installed = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing GRPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible GRPO dependency: expected {package}=={expected}, got {installed[package]}"
            )

    try:
        import torch
        import verl
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from shopping_grpo.verl_adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.verl_adapter.tools import ShopSimulatorTool
        from shopping_grpo.verl_compat import install_torch_padding_fallback
        from verl.tools.base_tool import BaseTool
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL 0.8 install: required AgentLoop/Tool APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    verl_source = Path(verl.__file__).resolve()
    if "agentic-grpo-longhorizon" in str(verl_source):
        raise SystemExit(f"reference veRL fork is shadowing pip veRL 0.8: {verl_source}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the GRPO environment")
    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or not issubclass(ShopSimulatorTool, BaseTool)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("veRL 0.8 built-in qwen3_coder parser is unavailable")
    install_torch_padding_fallback()
    print(
        "GRPO runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in installed.items())
        + f", source={verl_source}"
    )


if __name__ == "__main__":
    main()
