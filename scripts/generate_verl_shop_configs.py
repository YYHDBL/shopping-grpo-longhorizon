#!/usr/bin/env python3
"""由唯一的 SHOP_TOOL_SCHEMAS 生成 veRL 0.8 工具配置。"""

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


def parse_args():
    parser = argparse.ArgumentParser(description="生成 veRL 0.8 ShopSimulator tool JSON")
    parser.add_argument("--tool-output", type=Path, default=Path("configs/verl/shop_tools.json"))
    return parser.parse_args()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    _write(args.tool_output, build_tool_config())
    print(f"tools={args.tool_output}")


if __name__ == "__main__":
    main()
