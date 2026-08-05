#!/usr/bin/env python3
"""Reopen and verify every hash in a complete Feed CPU workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.feed.workflow import verify_cpu_mvp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_cpu_mvp(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
