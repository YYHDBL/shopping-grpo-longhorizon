#!/usr/bin/env python3
"""从 ShopSimulator 当前使用的目标生成器导出公开 task_id 清单。

必须用 ShopSimulator 自己的 Python 环境运行本脚本，避免在本仓库复制商品
清洗或 goal 构造逻辑，从而确保 task_id 与运行中的环境完全一致。
"""

import argparse
import json
import sys
from pathlib import Path


def load_goal_count(shopsim_root: Path) -> int:
    """加载与 ShopSimulator API 相同的数据和 goal 构造逻辑，返回任务数。"""
    shopsim_root = shopsim_root.resolve()
    if not (shopsim_root / "web_agent_site").is_dir():
        raise ValueError(
            f"--shopsim-root 必须指向 ShopSimulator/shop_env，未找到 web_agent_site：{shopsim_root}"
        )

    # ShopSimulator 不是安装包；仅在本次导出时把其根目录加入模块搜索路径。
    sys.path.insert(0, str(shopsim_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    from web_agent_site.utils import DEFAULT_FILE_PATH

    products, _, prices, _ = load_products(
        DEFAULT_FILE_PATH,
        num_products=None,
        human_goals=None,
    )
    return len(get_goals(products, prices))


def write_task_ids(goal_count: int, output_path: Path, force: bool = False) -> int:
    """按环境 goal 顺序写入连续 task_id，默认保护已有文件。"""
    if goal_count < 0:
        raise ValueError("goal_count 不能小于 0")
    if output_path.exists() and not force:
        raise FileExistsError(f"输出文件已存在：{output_path}；确认替换请加 --force")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for task_id in range(goal_count):
            output_file.write(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n")
    return goal_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出与 ShopSimulator 当前环境一致的 task_id 清单。")
    parser.add_argument(
        "--shopsim-root",
        type=Path,
        required=True,
        help="ShopSimulator 的 shop_env 目录，例如 ../ShopSimulator/shop_env",
    )
    parser.add_argument("--output", type=Path, default=Path("data/shop_tasks.jsonl"))
    parser.add_argument("--force", action="store_true", help="允许覆盖已有输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goal_count = load_goal_count(args.shopsim_root)
    written = write_task_ids(goal_count, args.output, force=args.force)
    print(f"已导出 {written} 个 task_id 到 {args.output}")


if __name__ == "__main__":
    main()
