#!/usr/bin/env python3
"""Launch the versioned Feed GRPO profile, or print its exact command safely."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.check_feed_grpo_runtime import validate_configs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "feed_grpo.yaml"
DEFAULT_CATALOG = (
    ROOT
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "data"
    / "fine_items_eval_train_all.json.gz"
)


# Hydra's CLI can replace arbitrary composed subtrees.  These paths define the
# validated Feed runtime/reward boundary or identify the audited inputs and
# output, so accepting a late override would make the preflight meaningless.
PROTECTED_HYDRA_PATHS = (
    "defaults",
    "hydra",
    "data.train_files",
    "data.val_files",
    "data.max_prompt_length",
    "data.max_response_length",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.mode",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.max_model_len",
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.agent",
    "actor_rollout_ref.rollout.multi_turn",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    "algorithm.adv_estimator",
    "algorithm.use_kl_in_reward",
    "shopping_dynamic_sampling",
    "feed_process_credit",
    "feed_runtime",
    "reward_model",
    "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook",
    "trainer.default_local_dir",
)


def _validated(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _model_has_weights(path: Path) -> bool:
    return any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="source five-artifact dataset; required for a real training launch",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--credit-mode",
        choices=("terminal", "rtg", "event", "counterfactual"),
        default="terminal",
    )
    parser.add_argument(
        "--acknowledge-metadata-only",
        action="store_true",
        help="allow non-terminal credit metadata while vanilla veRL still trains on terminal scalar",
    )
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument("--experiment-name", default="feed-longhorizon-grpo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def validate_hydra_overrides(overrides: list[str]) -> list[str]:
    """Reject late overrides that could bypass the audited Feed contract."""

    validated: list[str] = []
    for override in overrides:
        token = str(override).strip()
        if not token:
            raise SystemExit("empty Hydra override is not allowed")
        if token.startswith("--"):
            raise SystemExit(
                f"Hydra command-line option injection is not allowed: {token}"
            )
        raw_key = token.split("=", 1)[0].strip()
        key = raw_key.lstrip("+~")
        if not key:
            raise SystemExit(f"invalid Hydra override: {token}")
        # Hydra config-group syntax can use slashes and an ``@package`` target.
        # Check both the selected group and its destination to avoid bypassing a
        # protected dotted path through alternate spelling.
        group, separator, package = key.partition("@")
        candidates = {group.replace("/", ".")}
        if separator and package:
            candidates.add(package.replace("/", "."))
        if any(
            candidate == protected
            or candidate.startswith(protected + ".")
            or protected.startswith(candidate + ".")
            for candidate in candidates
            for protected in PROTECTED_HYDRA_PATHS
        ):
            raise SystemExit(
                "Hydra override would alter the validated Feed runtime contract: "
                + raw_key
            )
        validated.append(token)
    return validated


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str], dict[str, object]]:
    model = _validated(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file() or not _model_has_weights(model):
        raise SystemExit(f"model directory is incomplete: {model}")
    train_data = _validated(args.train_data, "train parquet")
    val_data = _validated(args.val_data, "validation parquet")
    if train_data.parent != val_data.parent:
        raise SystemExit("train and validation Parquet must share one manifest directory")
    data_manifest = train_data.parent / "manifest.json"
    dataset_dir = None
    if args.dataset_dir is not None:
        dataset_dir = _validated(args.dataset_dir, "Feed dataset directory")
        if not dataset_dir.is_dir() or not (dataset_dir / "manifest.json").is_file():
            raise SystemExit(f"Feed dataset directory is incomplete: {dataset_dir}")
    if not args.dry_run:
        if dataset_dir is None:
            raise SystemExit("a real Feed GRPO launch requires --dataset-dir")
        _validated(data_manifest, "Feed Parquet manifest")
    catalog = _validated(args.catalog, "catalog")
    config = _validated(args.config, "Feed GRPO config")
    config_report = validate_configs(ROOT, grpo_path=config)
    try:
        model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"model config.json is invalid: {model / 'config.json'}") from exc
    if not isinstance(model_config, dict):
        raise SystemExit("model config.json must contain an object")
    declared_context = next(
        (
            int(model_config[key])
            for key in ("max_position_embeddings", "seq_length", "model_max_length")
            if isinstance(model_config.get(key), int)
            and not isinstance(model_config.get(key), bool)
            and int(model_config[key]) > 0
        ),
        None,
    )
    required_context = int(config_report["context_tokens"])
    if declared_context is not None and declared_context < required_context:
        raise SystemExit(
            f"model context {declared_context} is smaller than Feed runtime {required_context}"
        )
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"output directory must be new or empty: {output}")
    if args.credit_mode != "terminal" and not args.acknowledge_metadata_only:
        raise SystemExit(
            "rtg/event/counterfactual are audit metadata in vanilla veRL 0.8; "
            "pass --acknowledge-metadata-only to train on terminal scalar anyway"
        )
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "FEED_GRPO_MODEL_PATH": str(model),
            "FEED_GRPO_TRAIN_FILE": str(train_data),
            "FEED_GRPO_VAL_FILE": str(val_data),
            "FEED_GRPO_OUTPUT_DIR": str(output),
            "FEED_CATALOG_PATH": str(catalog),
            "FEED_CREDIT_MODE": args.credit_mode,
        }
    )
    overrides = [
        f"trainer.logger=[{args.logger}]",
        f"trainer.experiment_name={args.experiment_name}",
    ]
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    extra = validate_hydra_overrides(extra)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *overrides,
        *extra,
    ]
    audit: dict[str, object] = {
        "training_started": False,
        "command": command,
        "model": str(model),
        "train_data": str(train_data),
        "validation_data": str(val_data),
        "data_manifest": str(data_manifest),
        "dataset_dir": None if dataset_dir is None else str(dataset_dir),
        "catalog": str(catalog),
        "output": str(output),
        "config": str(config),
        "credit_mode": args.credit_mode,
        "native_advantage_integration": False,
        "required_context_tokens": required_context,
        "model_declared_context_tokens": declared_context,
    }
    return command, environment, audit


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command, environment, audit = build_command(args)
    if args.dry_run:
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    preflight = [
        sys.executable,
        str(ROOT / "scripts" / "check_feed_grpo_runtime.py"),
        "--require-runtime",
        "--grpo-config",
        str(audit["config"]),
        "--train-data",
        str(audit["train_data"]),
        "--val-data",
        str(audit["validation_data"]),
        "--data-manifest",
        str(audit["data_manifest"]),
        "--dataset-dir",
        str(audit["dataset_dir"]),
    ]
    status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if status:
        raise SystemExit(status)
    Path(environment["FEED_GRPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    audit["training_started"] = True
    print(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        flush=True,
    )
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
