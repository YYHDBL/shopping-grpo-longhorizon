#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS


def build_tool_config():
    return {
        "tools": [
            {
                "class_name": f"shopping_grpo.verl_shop_tools.Shop_{schema['function']['name']}_Tool",
                "config": {"type": "native"},
                "tool_schema": schema,
            }
            for schema in SHOP_TOOL_SCHEMAS
        ]
    }


def write_tool_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_tool_config(), ensure_ascii=False, indent=2) + "\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate shopping veRL tool config.")
    parser.add_argument("--output", type=Path, default=Path("configs/tool_config/shop_tools.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    print(write_tool_config(args.output))


if __name__ == "__main__":
    main()
