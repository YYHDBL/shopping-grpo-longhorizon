#!/usr/bin/env python3
"""由唯一的 SHOP_TOOL_SCHEMAS 生成 veRL 工具与交互配置，避免双份 schema 漂移。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS


def build_tool_config():
    return {
        "tools": [
            {
                "class_name": "shopping_grpo.verl_adapter.tools.ShopSimulatorTool",
                "config": {"type": "native"},
                "tool_schema": schema,
            }
            for schema in SHOP_TOOL_SCHEMAS
        ]
    }


def build_interaction_config(base_url: str, max_steps: int):
    return {
        "interaction": [
            {
                "name": "shopsimulator",
                "class_name": "shopping_grpo.verl_adapter.interaction.ShopSimulatorInteraction",
                "config": {"base_url": base_url, "timeout": 60, "max_steps": int(max_steps)},
            }
        ]
    }


def parse_args():
    parser = argparse.ArgumentParser(description="生成 veRL ShopSimulator JSON 配置")
    parser.add_argument("--tool-output", type=Path, default=Path("configs/verl/shop_tools.json"))
    parser.add_argument("--interaction-output", type=Path, default=Path("configs/verl/shop_interaction.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--max-steps", type=int, default=35)
    return parser.parse_args()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    _write(args.tool_output, build_tool_config())
    _write(args.interaction_output, build_interaction_config(args.base_url, args.max_steps))
    print(f"tools={args.tool_output} interaction={args.interaction_output}")


if __name__ == "__main__":
    main()
