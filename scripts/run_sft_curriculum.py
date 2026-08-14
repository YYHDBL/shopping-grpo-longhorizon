#!/usr/bin/env python3
"""Run the fixed Pure V4 SFT curriculum with existing train/merge scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("a", "b", "c")


def build_stage_commands(
    manifest,
    *,
    manifest_path,
    source,
    base_model,
    output_root,
    python,
    start_stage="a",
    stop_after_stage="c",
    swanlab=False,
    swanlab_project="shopping-grpo-sft-curriculum",
    qlora=False,
    liger_kernel=False,
    resume_from_checkpoint=None,
):
    start = STAGES.index(start_stage)
    stop = STAGES.index(stop_after_stage)
    if stop < start:
        raise ValueError("--stop-after-stage must be at or after --start-stage")

    commands = []
    for index, stage in enumerate(STAGES[start : stop + 1]):
        stage_config = manifest["stages"][stage]
        stage_root = Path(output_root) / f"stage-{stage}"
        model = (
            base_model
            if stage == "a"
            else str(Path(output_root) / f"stage-{STAGES[STAGES.index(stage) - 1]}" / "merged")
        )
        train = [
            str(python),
            str(ROOT / "scripts/train_lora_sft.py"),
            "--model",
            str(model),
            "--train",
            str(source),
            "--validation",
            str(source),
            "--curriculum-manifest",
            str(manifest_path),
            "--curriculum-stage",
            stage,
            "--output",
            str(stage_root / "adapter"),
            "--epochs",
            str(stage_config["epochs"]),
            "--learning-rate",
            str(stage_config["learning_rate"]),
            "--max-length",
            "24576",
            "--dtype",
            "bf16",
            "--attention-implementation",
            "sdpa",
            "--gradient-checkpointing",
            "--swanlab-run-name",
            f"pure-v4-stage-{stage}",
        ]
        if swanlab:
            train.extend(["--swanlab", "--swanlab-project", swanlab_project])
        if qlora:
            train.append("--qlora")
        if liger_kernel:
            train.append("--liger-kernel")
        if index == 0 and resume_from_checkpoint:
            train.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
        merge = [
            str(python),
            str(ROOT / "scripts/merge_lora_adapter.py"),
            "--base-model",
            str(model),
            "--adapter",
            str(stage_root / "adapter"),
            "--output",
            str(stage_root / "merged"),
            "--bf16",
        ]
        commands.append({"stage": stage, "train": train, "merge": merge})
    return commands


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/sft_pure_v4/all.jsonl"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/sft_curriculum/manifest.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/models/sft-curriculum"
    )
    parser.add_argument("--start-stage", choices=STAGES, default="a")
    parser.add_argument("--stop-after-stage", choices=STAGES, default="c")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="shopping-grpo-sft-curriculum")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--liger-kernel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shopping-sft-curriculum-v1":
        raise SystemExit("不支持的课程清单 schema_version")
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_sha != manifest.get("source", {}).get("sha256"):
        raise SystemExit("Pure V4 数据与课程清单 SHA256 不一致；请先重新生成并审查清单")
    try:
        commands = build_stage_commands(
            manifest,
            manifest_path=args.manifest,
            source=args.source,
            base_model=args.base_model,
            output_root=args.output_root,
            python=sys.executable,
            start_stage=args.start_stage,
            stop_after_stage=args.stop_after_stage,
            swanlab=args.swanlab,
            swanlab_project=args.swanlab_project,
            qlora=args.qlora,
            liger_kernel=args.liger_kernel,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for command in commands:
        print(f"\n[stage {command['stage'].upper()}] train")
        print(shlex.join(command["train"]))
        print(f"[stage {command['stage'].upper()}] merge")
        print(shlex.join(command["merge"]))
        if args.dry_run:
            continue
        merged = args.output_root / f"stage-{command['stage']}" / "merged"
        if merged.exists() and any(merged.iterdir()):
            raise SystemExit(f"拒绝覆盖已完成阶段：{merged}")
        subprocess.run(command["train"], check=True)
        subprocess.run(command["merge"], check=True)
    last_stage = commands[-1]["stage"]
    label = "最终 GRPO 起点" if last_stage == "c" else "本次最后阶段输出"
    print(f"\n{label}：{args.output_root / f'stage-{last_stage}/merged'}")


if __name__ == "__main__":
    main()
