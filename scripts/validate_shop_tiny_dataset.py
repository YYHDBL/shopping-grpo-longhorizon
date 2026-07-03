#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def validate(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["extra_info"]["index"] != len(rows):
            raise ValueError(f"line {line_no}: index must be {len(rows)}")
        if row["extra_info"]["interaction_kwargs"]["name"] != "shop":
            raise ValueError(f"line {line_no}: interaction name must be shop")
        if "task_id" not in row["extra_info"]["interaction_kwargs"]:
            raise ValueError(f"line {line_no}: missing task_id")
        if not row.get("prompt") or row["prompt"][0]["role"] != "system":
            raise ValueError(f"line {line_no}: prompt must start with system")
        rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/shop_tiny_tasks.jsonl")
    rows = validate(path)
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
