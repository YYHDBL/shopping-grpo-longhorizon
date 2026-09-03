#!/usr/bin/env python3
"""Resolve a named SFT/GRPO ablation into the repository's existing launcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs/experiments.json"
TARGET_MODULE_PRESETS = {
    "attention_only": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "attention_mlp": (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ),
    "full": (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    ),
}


def load_registry(path: Path) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("version") != 1:
        raise ValueError("experiment registry version must equal 1")
    if not isinstance(registry.get("baselines"), dict) or not isinstance(
        registry.get("experiments"), dict
    ):
        raise ValueError("experiment registry requires baselines and experiments objects")
    return registry


def _decode_override(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise ValueError(f"override must use key=value: {raw!r}")
    key, value = raw.split("=", 1)
    if not key:
        raise ValueError("override key cannot be empty")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def resolve_experiment(registry: dict, name: str, cli_overrides=()) -> dict:
    try:
        definition = registry["experiments"][name]
        stage = definition["stage"]
        settings = deepcopy(registry["baselines"][stage])
    except KeyError as exc:
        raise ValueError(f"unknown or malformed experiment: {name}") from exc
    if stage not in {"sft", "grpo"}:
        raise ValueError(f"unsupported experiment stage: {stage!r}")
    overrides = definition.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"experiment overrides must be an object: {name}")
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError(f"unknown {stage} settings: {sorted(unknown)}")
    settings.update(overrides)
    for raw in cli_overrides:
        key, value = _decode_override(raw)
        if key not in settings:
            raise ValueError(f"unknown {stage} setting: {key}")
        settings[key] = value
    _validate_settings(stage, settings)
    return {"name": name, "stage": stage, "settings": settings}


def _positive(settings: dict, *keys: str) -> None:
    for key in keys:
        if float(settings[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def _validate_settings(stage: str, settings: dict) -> None:
    if stage == "sft":
        _positive(settings, "learning_rate", "epochs", "lora_rank", "lora_alpha", "subset_seed")
        if settings["target_modules"] not in TARGET_MODULE_PRESETS:
            raise ValueError(f"unknown target_modules preset: {settings['target_modules']!r}")
        if settings["train_count"] is not None and settings["train_ratio"] is not None:
            raise ValueError("train_count and train_ratio are mutually exclusive")
        if settings["train_count"] is not None and int(settings["train_count"]) <= 0:
            raise ValueError("train_count must be positive")
        if settings["train_ratio"] is not None and not 0 < float(settings["train_ratio"]) <= 1:
            raise ValueError("train_ratio must be in (0, 1]")
        return
    _positive(
        settings,
        "learning_rate",
        "rollout_number",
        "prompt_batch_size",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "soft_length_threshold",
        "max_environment_steps",
    )
    if not settings["dynamic_sampling"]:
        raise ValueError("dynamic_sampling is fixed on for this repository")
    if int(settings["max_environment_steps"]) != 35:
        raise ValueError("max_environment_steps is fixed at 35 by the environment contract")
    if int(settings["ppo_mini_batch_size"]) % int(settings["ppo_micro_batch_size"]):
        raise ValueError("ppo_mini_batch_size must be divisible by ppo_micro_batch_size")
    if settings["clip_mode"] not in {"symmetric", "clip_higher"}:
        raise ValueError("clip_mode must be symmetric or clip_higher")
    if min(float(settings["kl_coefficient"]), float(settings["length_penalty_per_step"]), float(settings["max_length_penalty"])) < 0:
        raise ValueError("KL and length penalty coefficients must be non-negative")


def _python(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    return str(candidate if candidate.is_file() else Path(sys.executable))


def build_experiment(
    experiment: dict,
    *,
    root: Path = ROOT,
    output_root: Path = Path("outputs/ablations"),
    model: str | Path | None = None,
    train_data: Path | None = None,
    validation_data: Path | None = None,
) -> tuple[list[str], dict[str, str], Path]:
    root = Path(root).resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    output = output_root / experiment["name"]
    settings = experiment["settings"]
    environment = dict(os.environ)
    source_path = str(root / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_path, existing_python_path))
        if existing_python_path
        else source_path
    )
    if experiment["stage"] == "sft":
        command = [
            _python(root),
            "scripts/train_lora_sft.py",
            "--model", str(model or "Qwen/Qwen3.5-2B"),
            "--train", str(train_data or root / "data/sft/train.jsonl"),
            "--validation", str(validation_data or root / "data/sft/validation.jsonl"),
            "--output", str(output),
            "--learning-rate", str(settings["learning_rate"]),
            "--epochs", str(settings["epochs"]),
            "--lora-r", str(settings["lora_rank"]),
            "--lora-alpha", str(settings["lora_alpha"]),
            "--target-modules", *TARGET_MODULE_PRESETS[settings["target_modules"]],
            "--subset-seed", str(settings["subset_seed"]),
            "--save-total-limit", str(settings["save_total_limit"]),
            "--dtype", "auto",
            "--gradient-checkpointing",
            "--attention-implementation", "sdpa",
        ]
        if settings["train_count"] is not None:
            command.extend(("--train-count", str(settings["train_count"])))
        if settings["train_ratio"] is not None:
            command.extend(("--train-ratio", str(settings["train_ratio"])))
        return command, environment, output

    low, high = (0.2, 0.2) if settings["clip_mode"] == "symmetric" else (0.2, 0.28)
    environment.update(
        {
            "SHOPPING_LENGTH_SHAPING_ENABLE": str(bool(settings["length_shaping_enabled"])).lower(),
            "SHOPPING_SOFT_LENGTH_THRESHOLD": str(settings["soft_length_threshold"]),
            "SHOPPING_LENGTH_PENALTY_PER_STEP": str(settings["length_penalty_per_step"]),
            "SHOPPING_MAX_LENGTH_PENALTY": str(settings["max_length_penalty"]),
        }
    )
    command = [
        _python(root),
        "scripts/train_grpo.py",
        "--model", str(model or root / "outputs/models/sft-merged"),
        "--train-data", str(train_data or root / "data/grpo/train.parquet"),
        "--val-data", str(validation_data or root / "data/grpo/validation.parquet"),
        "--output", str(output),
        "--experiment-name", experiment["name"],
        "--",
        f"actor_rollout_ref.actor.optim.lr={settings['learning_rate']}",
        f"actor_rollout_ref.rollout.n={settings['rollout_number']}",
        f"data.train_batch_size={settings['prompt_batch_size']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={settings['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={settings['ppo_micro_batch_size']}",
        f"actor_rollout_ref.actor.clip_ratio_low={low}",
        f"actor_rollout_ref.actor.clip_ratio_high={high}",
        f"actor_rollout_ref.actor.use_kl_loss={str(bool(settings['kl_enabled'])).lower()}",
        f"actor_rollout_ref.actor.kl_loss_coef={settings['kl_coefficient']}",
        f"shopping_trace.enable={str(bool(settings['trace_enabled'])).lower()}",
    ]
    return command, environment, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name")
    parser.add_argument("--config", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--model")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = resolve_experiment(
        load_registry(args.config), args.experiment_name, args.set
    )
    command, environment, output = build_experiment(
        experiment,
        output_root=args.output_root,
        model=args.model,
        train_data=args.train_data,
        validation_data=args.validation_data,
    )
    print(json.dumps({"experiment": experiment, "command": command, "output": str(output)}, indent=2))
    if args.dry_run:
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"experiment output directory must be new or empty: {output}")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
