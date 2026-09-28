#!/usr/bin/env python3
"""V2.0 数据验收管线入口。

用法：
    python scripts/run_acceptance.py --limit 50        # 试跑 50 条（两级）
    python scripts/run_acceptance.py --dry-run          # 只跑第一级（本地、零成本）
    python scripts/run_acceptance.py                    # 全量 23,421 条

Jev key 从环境变量 OPENROUTER_API_KEY 读取，默认从仓库根目录 .env 加载。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import JevDecisionsClient  # noqa: E402
from shopping_grpo.acceptance.pipeline import run  # noqa: E402

DEFAULT_DATA = (
    ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
)
DEFAULT_OUT = ROOT / "outputs/acceptance"


def load_env_file(path: Path) -> None:
    """把 .env 里的键值注入 os.environ；已存在的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument("--dry-run", action="store_true", help="只跑第一级硬约束")
    parser.add_argument("--workers", type=int, default=8, help="Jev 并发数")
    parser.add_argument("--model", default=None, help="覆盖 jev 模型 ID")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = None
    if not args.dry_run:
        if not api_key:
            print("错误：未找到 OPENROUTER_API_KEY（.env 或环境变量）", file=sys.stderr)
            return 2
        client = JevDecisionsClient(api_key=api_key, model=args.model)

    def progress(done, total):
        print(f"jev 进度 {done}/{total}", flush=True)

    report = run(
        args.data,
        args.out,
        client=client,
        limit=args.limit,
        dry_run=args.dry_run,
        workers=args.workers,
        progress=progress,
    )
    print(json.dumps(report["verdict_counts"], ensure_ascii=False, indent=2))
    print(f"报告：{Path(args.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
