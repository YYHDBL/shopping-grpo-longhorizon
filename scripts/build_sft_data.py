#!/usr/bin/env python3
import argparse
from pathlib import Path

from shopping_grpo.sft_data import process_raw_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Build accepted trajectories and SFT JSONL.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, default=Path("outputs/sft/accepted_trajectories.jsonl"))
    parser.add_argument("--rejected", type=Path, default=Path("outputs/sft/rejected_trajectories.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("outputs/sft/reject_stats.json"))
    parser.add_argument("--sft", type=Path, default=Path("outputs/sft/openai_messages.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    summary = process_raw_trajectories(
        raw_path=args.raw,
        accepted_path=args.accepted,
        rejected_path=args.rejected,
        stats_path=args.stats,
        sft_path=args.sft,
    )
    print(summary)


if __name__ == "__main__":
    main()
