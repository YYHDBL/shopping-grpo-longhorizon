#!/usr/bin/env python3
"""Static preflight for the Feed veRL 0.8 profile.

This command reads configuration, imports the adapter, and checks the installed
veRL version/API when present.  It never loads a model, initializes Ray/CUDA,
opens an environment episode, or starts training.

Missing veRL is allowed by default so CPU-only artifact builders can run.  Pass
``--require-runtime`` on a training node to make a missing/incomplete pinned
runtime an error.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import source_from_cache
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_TOOL_NAMES = {
    "retrieve_products",
    "inspect_product",
    "compare_products",
    "read_reviews",
    "find_alternatives",
    "find_complements",
    "check_inventory",
    "commit_recommendation",
}
ADAPTER_CLASS = "shopping_grpo.feed.verl_adapter.FeedShoppingTool"
LOOP_CLASS = "shopping_grpo.feed.verl_adapter.FeedToolAgentLoop"
EXPECTED_TOOL_PAYLOAD_VERSION = "feed-tool-delta-v1"
EXPECTED_SHOPPING_PATCH_COMPATIBILITY = "feed-shopping-extra-v1"
DYNAMIC_SAMPLING_PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3"
STRICT_TOOL_SCHEMA_PATCH_MARKER = "SHOPPING_GRPO_STRICT_TOOL_SCHEMA_PATCH_V1"


_YAML_KEY = re.compile(
    r"^(?P<indent>\s*)(?:-\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_.-]*):(?:\s*(?P<value>.*))?$"
)


def extract_yaml_scalars(text: str) -> dict[str, Any]:
    """Extract scalar values by dotted indentation path without a YAML dependency."""

    values: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-") and ":" not in stripped:
            continue
        match = _YAML_KEY.match(raw_line)
        if match is None:
            continue
        indent = len(match.group("indent").replace("\t", "    "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        key = match.group("key")
        raw_value = (match.group("value") or "").strip()
        path = ".".join([item[1] for item in stack] + [key])
        if raw_value:
            values[path] = _parse_scalar(raw_value)
        else:
            stack.append((indent, key))
    return values


def validate_configs(
    root: Path,
    *,
    grpo_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the frozen Feed tool/AgentLoop/GRPO contracts."""

    config_dir = root / "configs"
    tool_path = config_dir / "feed_tools.json"
    agent_path = config_dir / "feed_agent_loop.yaml"
    grpo_path = (
        grpo_path.expanduser().resolve()
        if grpo_path is not None
        else (config_dir / "feed_grpo.yaml").resolve()
    )
    for path in (tool_path, agent_path, grpo_path):
        if not path.is_file():
            raise ValueError(f"missing Feed runtime config: {path}")

    tools = json.loads(tool_path.read_text(encoding="utf-8")).get("tools")
    if not isinstance(tools, list):
        raise ValueError("feed_tools.json must contain a tools array")
    names = {
        item.get("tool_schema", {}).get("function", {}).get("name")
        for item in tools
        if isinstance(item, Mapping)
    }
    if names != EXPECTED_TOOL_NAMES:
        raise ValueError(
            "Feed tool set mismatch: "
            + json.dumps(
                {
                    "missing": sorted(EXPECTED_TOOL_NAMES - names),
                    "extra": sorted(names - EXPECTED_TOOL_NAMES),
                },
                sort_keys=True,
            )
        )
    if any(item.get("class_name") != ADAPTER_CLASS for item in tools):
        raise ValueError("every Feed tool must use FeedShoppingTool")

    agent_values = extract_yaml_scalars(agent_path.read_text(encoding="utf-8"))
    grpo_values = extract_yaml_scalars(grpo_path.read_text(encoding="utf-8"))
    _expect(agent_values, "name", "feed_tool_agent")
    _expect(agent_values, "_target_", LOOP_CLASS)
    _expect(agent_values, "min_feed_steps", 24)
    _expect(agent_values, "max_feed_steps", 48)
    _expect(agent_values, "max_info_calls_per_video", 3)
    _expect(agent_values, "required_environment_version", "feed-environment-v1")
    _expect(agent_values, "required_observation_version", "feed-observation-v1")
    _expect(agent_values, "required_tools_version", "feed-tools-v1")
    _expect(agent_values, "required_reward_version", "feed-reward-v1")
    _expect(
        agent_values,
        "tool_payload_version",
        EXPECTED_TOOL_PAYLOAD_VERSION,
    )
    if int(agent_values.get("expected_min_turn_capacity", 0)) < 128:
        raise ValueError("Feed AgentLoop must declare capacity for at least 128 turns")
    _expect(agent_values, "process_credit_metadata_only", True)
    _expect(agent_values, "native_advantage_integration", False)
    credit_default = _omegaconf_default(agent_values.get("credit_mode"))
    if credit_default not in {"terminal", "rtg", "event", "counterfactual"}:
        raise ValueError("Feed AgentLoop has an invalid default credit_mode")

    _expect(grpo_values, "algorithm.adv_estimator", "grpo")
    _expect(grpo_values, "actor_rollout_ref.rollout.mode", "async")
    if int(grpo_values.get("actor_rollout_ref.rollout.n", 0)) < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    _expect(grpo_values, "actor_rollout_ref.rollout.multi_turn.enable", True)
    _expect(grpo_values, "actor_rollout_ref.rollout.multi_turn.max_parallel_calls", 1)
    if int(grpo_values.get("actor_rollout_ref.rollout.multi_turn.max_user_turns", 0)) < 128:
        raise ValueError("max_user_turns must be at least 128")
    if int(grpo_values.get("actor_rollout_ref.rollout.multi_turn.max_assistant_turns", 0)) < 128:
        raise ValueError("max_assistant_turns must be at least 128")
    tool_config = str(
        grpo_values.get("actor_rollout_ref.rollout.multi_turn.tool_config_path", "")
    )
    if not tool_config.endswith("/configs/feed_tools.json"):
        raise ValueError("GRPO multi_turn.tool_config_path must select feed_tools.json")
    loop_config = str(
        grpo_values.get("actor_rollout_ref.rollout.agent.agent_loop_config_path", "")
    )
    if not loop_config.endswith("/configs/feed_agent_loop.yaml"):
        raise ValueError("GRPO agent_loop_config_path must select feed_agent_loop.yaml")
    _expect(grpo_values, "actor_rollout_ref.rollout.agent.default_agent_loop", "feed_tool_agent")
    _expect(grpo_values, "shopping_dynamic_sampling.enable", True)
    _expect(grpo_values, "shopping_dynamic_sampling.metric", "seq_reward")
    _expect(grpo_values, "reward_model.enable", False)
    _expect(grpo_values, "feed_runtime.llm_judge", False)
    _expect(grpo_values, "feed_process_credit.metadata_only", True)
    _expect(grpo_values, "feed_process_credit.native_advantage_integration", False)
    _expect(
        grpo_values,
        "feed_process_credit.native_reward_boundary",
        "scalar_terminal_reward_only",
    )
    _expect(grpo_values, "feed_runtime.required_verl_version", EXPECTED_VERL_VERSION)

    prompt = int(grpo_values.get("data.max_prompt_length", 0))
    response = int(grpo_values.get("data.max_response_length", 0))
    model_length = int(grpo_values.get("actor_rollout_ref.rollout.max_model_len", 0))
    if prompt <= 0 or response < 8192 or prompt + response != model_length:
        raise ValueError("Feed context lengths are missing or inconsistent")
    if int(agent_values.get("context_window_tokens", 0)) != model_length:
        raise ValueError("Feed AgentLoop context_window_tokens must equal rollout.max_model_len")
    for key in (
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    ):
        if int(grpo_values.get(key, 0)) != model_length:
            raise ValueError(f"{key} must equal rollout.max_model_len")

    return {
        "tool_count": len(tools),
        "information_tool_count": len(names - {"commit_recommendation"}),
        "max_feed_steps": int(agent_values["max_feed_steps"]),
        "min_feed_steps": int(agent_values["min_feed_steps"]),
        "max_user_turns": int(
            grpo_values["actor_rollout_ref.rollout.multi_turn.max_user_turns"]
        ),
        "max_assistant_turns": int(
            grpo_values["actor_rollout_ref.rollout.multi_turn.max_assistant_turns"]
        ),
        "context_tokens": model_length,
        "credit_mode_default": credit_default,
        "grpo_config": str(grpo_path),
        "tool_payload_version": EXPECTED_TOOL_PAYLOAD_VERSION,
        "dynamic_sampling": True,
        "llm_judge": False,
    }


def inspect_runtime(*, require_runtime: bool) -> dict[str, Any]:
    """Import the fallback adapter and, when installed, veRL 0.8 APIs."""

    adapter = import_module("shopping_grpo.feed.verl_adapter")
    for name in ("FeedShoppingTool", "FeedToolAgentLoop", "build_process_credit_metadata"):
        if not hasattr(adapter, name):
            raise ValueError(f"Feed adapter is missing {name}")
    _expect_adapter_constant(
        adapter,
        "FEED_TOOL_PAYLOAD_VERSION",
        EXPECTED_TOOL_PAYLOAD_VERSION,
    )
    _expect_adapter_constant(
        adapter,
        "SHOPPING_PATCH_COMPATIBILITY",
        EXPECTED_SHOPPING_PATCH_COMPATIBILITY,
    )

    try:
        installed = version("verl")
    except PackageNotFoundError:
        if require_runtime:
            raise ValueError(f"missing required runtime: verl=={EXPECTED_VERL_VERSION}")
        return {
            "verl": None,
            "runtime_ready": False,
            "adapter_mode": "dependency_free_fallback",
            "tool_payload_version": EXPECTED_TOOL_PAYLOAD_VERSION,
            "shopping_patch_compatibility": EXPECTED_SHOPPING_PATCH_COMPATIBILITY,
        }
    normalized = installed.split("+", 1)[0]
    if normalized != EXPECTED_VERL_VERSION:
        raise ValueError(
            f"incompatible veRL: expected {EXPECTED_VERL_VERSION}, got {installed}"
        )
    try:
        tool_module = import_module("verl.experimental.agent_loop.tool_agent_loop")
        base_module = import_module("verl.tools.base_tool")
        schema_module = import_module("verl.tools.schemas")
        trainer_module = import_module("verl.trainer.ppo.ray_trainer")
    except ImportError as exc:
        raise ValueError(f"veRL 0.8 ToolAgentLoop API is unavailable: {exc}") from exc
    tool_loop = getattr(tool_module, "ToolAgentLoop", None)
    agent_state = getattr(tool_module, "AgentState", None)
    base_tool = getattr(base_module, "BaseTool", None)
    if tool_loop is None or agent_state is None or base_tool is None:
        raise ValueError("veRL 0.8 ToolAgentLoop/BaseTool symbols are incomplete")
    if not issubclass(adapter.FeedToolAgentLoop, tool_loop):
        raise ValueError("FeedToolAgentLoop does not inherit the installed ToolAgentLoop")
    if not issubclass(adapter.FeedShoppingTool, base_tool):
        raise ValueError("FeedShoppingTool does not inherit the installed BaseTool")
    if getattr(getattr(agent_state, "TERMINATED", None), "value", None) != "terminated":
        raise ValueError("veRL AgentState lifecycle is incompatible")
    if not hasattr(tool_loop, "_handle_processing_tools_state"):
        raise ValueError("veRL ToolAgentLoop lifecycle hook is unavailable")
    schema_class = getattr(schema_module, "OpenAIFunctionToolSchema", None)
    if schema_class is None:
        raise ValueError("veRL OpenAIFunctionToolSchema is unavailable")
    schema_roundtrip = validate_tool_schema_roundtrip(schema_class)
    patch_source = verify_dynamic_sampling_patch(trainer_module)
    schema_patch_source = verify_source_marker(
        schema_module,
        marker=STRICT_TOOL_SCHEMA_PATCH_MARKER,
        label="veRL tool schema",
    )
    return {
        "verl": installed,
        "runtime_ready": True,
        "adapter_mode": "verl_native",
        "tool_schema_roundtrip": schema_roundtrip,
        "dynamic_sampling_patch": {
            "marker": DYNAMIC_SAMPLING_PATCH_MARKER,
            "source": patch_source,
        },
        "strict_tool_schema_patch": {
            "marker": STRICT_TOOL_SCHEMA_PATCH_MARKER,
            "source": schema_patch_source,
        },
        "shopping_patch_compatibility": EXPECTED_SHOPPING_PATCH_COMPATIBILITY,
    }


def static_preflight(
    root: Path,
    *,
    require_runtime: bool = False,
    grpo_config: Path | None = None,
    train_data: Path | None = None,
    val_data: Path | None = None,
    data_manifest: Path | None = None,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all non-mutating checks and return a machine-readable report."""

    root = root.resolve()
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    config = validate_configs(root, grpo_path=grpo_config)
    runtime = inspect_runtime(require_runtime=require_runtime)
    data = validate_parquet_inputs(
        train_data,
        val_data,
        data_manifest=data_manifest,
        dataset_dir=dataset_dir,
    )
    return {
        "ok": True,
        "training_started": False,
        "root": str(root),
        "config": config,
        "runtime": runtime,
        "data": data,
        "capability_boundary": {
            "reward_score": "scalar terminal episode return",
            "process_credit": "extra_fields metadata only",
            "native_advantage_integration": False,
            "requires_trainer_patch_for_process_advantage": True,
        },
    }


def validate_parquet_inputs(
    train_data: Path | None,
    val_data: Path | None,
    *,
    data_manifest: Path | None = None,
    dataset_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Open and validate the exact Feed Parquet files passed to the launcher."""

    if train_data is None and val_data is None:
        return None
    if train_data is None or val_data is None:
        raise ValueError("both train_data and val_data are required for Parquet preflight")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("Feed Parquet preflight requires pyarrow") from exc

    reports: dict[str, Any] = {}
    identities: dict[str, set[tuple[str, str]]] = {}
    for split, raw_path in (("train", train_data), ("validation", val_data)):
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Feed {split} Parquet is missing: {path}")
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise ValueError(
                f"cannot open Feed {split} Parquet: {exc.__class__.__name__}"
            ) from exc
        required = {
            "data_source",
            "task_id",
            "prompt",
            "episode_seed",
            "catalog_products",
        }
        missing = sorted(required - set(table.column_names))
        if missing:
            raise ValueError(
                f"Feed {split} Parquet is missing columns: {', '.join(missing)}"
            )
        forbidden_columns = {"feed_task", "catalog", "catalog_path"}
        forbidden = sorted(forbidden_columns & set(table.column_names))
        if forbidden:
            raise ValueError(
                f"Feed {split} Parquet contains runtime override columns: "
                + ", ".join(forbidden)
            )
        rows = table.to_pylist()
        if not rows:
            raise ValueError(f"Feed {split} Parquet is empty")
        identities[split] = _validate_feed_parquet_rows(rows, split=split)
        reports[split] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": _sha256(path),
            "task_content_sha256": _feed_task_content_sha256(rows),
        }
    train_episodes = {episode_id for episode_id, _ in identities["train"]}
    validation_episodes = {episode_id for episode_id, _ in identities["validation"]}
    train_personas = {persona_id for _, persona_id in identities["train"]}
    validation_personas = {persona_id for _, persona_id in identities["validation"]}
    if train_episodes & validation_episodes:
        raise ValueError("Feed train/validation Parquet episode IDs overlap")
    if train_personas & validation_personas:
        raise ValueError("Feed train/validation Parquet persona IDs overlap")
    if data_manifest is not None:
        _validate_parquet_manifest(
            data_manifest,
            reports=reports,
        )
    if dataset_dir is not None:
        _validate_dataset_binding(
            dataset_dir,
            identities=identities,
            reports=reports,
            data_manifest=data_manifest,
        )
    return reports


def _validate_feed_parquet_rows(
    rows: list[Mapping[str, Any]],
    *,
    split: str,
) -> set[tuple[str, str]]:
    from shopping_grpo.feed.catalog import ProductCatalog
    from shopping_grpo.feed.observation import (
        parse_feed_observation,
        render_feed_observation,
    )
    from shopping_grpo.feed.schema import EpisodeSeed

    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if row.get("data_source") != "feed-shopping-v1":
            raise ValueError(f"Feed {split} row {index} has wrong data_source")
        task_id = str(row.get("task_id") or "")
        prompt = row.get("prompt")
        if not task_id or not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Feed {split} row {index} has invalid task_id/prompt")
        try:
            seed = json.loads(row["episode_seed"])
            products = json.loads(row["catalog_products"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Feed {split} row {index} has invalid JSON-encoded task truth"
            ) from exc
        if not isinstance(seed, Mapping) or not isinstance(products, list) or not products:
            raise ValueError(f"Feed {split} row {index} has invalid task truth shapes")
        extra = row.get("extra_info")
        if extra is not None and not isinstance(extra, Mapping):
            raise ValueError(f"Feed {split} row {index} has invalid extra_info")
        extra = dict(extra or {})
        forbidden_extra = {
            "feed_task",
            "catalog",
            "catalog_products",
            "catalog_path",
            "calibration",
        }
        hidden_overrides = sorted(forbidden_extra & set(extra))
        if hidden_overrides:
            raise ValueError(
                f"Feed {split} row {index} extra_info contains runtime overrides: "
                + ", ".join(hidden_overrides)
            )
        duplicate_seed = extra.get("episode_seed")
        if duplicate_seed is not None:
            try:
                duplicate_seed = (
                    json.loads(duplicate_seed)
                    if isinstance(duplicate_seed, str)
                    else duplicate_seed
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Feed {split} row {index} has invalid duplicate episode_seed"
                ) from exc
            if duplicate_seed != seed:
                raise ValueError(
                    f"Feed {split} row {index} duplicate episode_seed mismatch"
                )
        try:
            typed_seed = EpisodeSeed.from_dict(seed)
            catalog = ProductCatalog(products)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feed {split} row {index} has invalid seed/catalog contract"
            ) from exc
        episode_id = typed_seed.episode_id
        persona_id = typed_seed.persona.persona_id
        videos = typed_seed.videos
        metadata = typed_seed.metadata
        if episode_id != task_id or not persona_id:
            raise ValueError(f"Feed {split} row {index} has inconsistent identities")
        for key, expected in (
            ("episode_id", episode_id),
            ("task_id", task_id),
            ("persona_id", persona_id),
            ("split", split),
        ):
            if key in extra and str(extra[key]) != expected:
                raise ValueError(
                    f"Feed {split} row {index} extra_info {key} mismatch"
                )
        if not 24 <= len(videos) <= 48:
            raise ValueError(f"Feed {split} row {index} must contain 24--48 videos")
        if isinstance(metadata, Mapping) and metadata.get("split") not in {None, split}:
            raise ValueError(f"Feed {split} row {index} declares a different split")
        missing_products = [
            product_id
            for product_id in typed_seed.product_ids
            if catalog.get(product_id) is None
        ]
        if missing_products:
            raise ValueError(f"Feed {split} row {index} catalog misses seed products")
        try:
            initial_user = next(
                message
                for message in prompt
                if isinstance(message, Mapping) and message.get("role") == "user"
            )
            initial_observation = parse_feed_observation(initial_user["content"])
            render_feed_observation(initial_observation)
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ValueError(
                f"Feed {split} row {index} has invalid initial observation prompt"
            ) from exc
        if initial_observation.get("episode_id") != episode_id:
            raise ValueError(f"Feed {split} row {index} prompt identity mismatch")
        initial_persona = initial_observation.get("persona") or {}
        if (
            not isinstance(initial_persona, Mapping)
            or initial_persona.get("persona_id") != persona_id
        ):
            raise ValueError(f"Feed {split} row {index} prompt persona mismatch")
        if (
            initial_observation.get("step") != 0
            or initial_observation.get("total_steps") != len(videos)
            or initial_observation.get("done") is not False
        ):
            raise ValueError(f"Feed {split} row {index} prompt is not an initial state")
        identity = (episode_id, persona_id)
        if identity in identities:
            raise ValueError(f"Feed {split} Parquet contains duplicate task identities")
        identities.add(identity)
    return identities


def _feed_task_content_sha256(rows: list[Mapping[str, Any]]) -> str:
    normalized = []
    for row in rows:
        projected: dict[str, Any] = {}
        for key in (
            "data_source",
            "task_id",
            "prompt",
            "episode_seed",
            "catalog_products",
            "calibration",
        ):
            if key not in row or row[key] is None:
                continue
            value = row[key]
            if key in {"episode_seed", "catalog_products", "calibration"} and isinstance(
                value, str
            ):
                value = json.loads(value)
            projected[key] = value
        normalized.append(projected)
    normalized.sort(key=lambda item: str(item.get("task_id") or ""))
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_parquet_manifest(
    manifest_path: Path,
    *,
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    path = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Feed Parquet manifest is missing or invalid") from exc
    if manifest.get("schema_version") != "feed-grpo-parquet-manifest-v1":
        raise ValueError("Feed Parquet manifest schema mismatch")
    if manifest.get("inspect_only") is not False:
        raise ValueError("Feed training requires a materialized Parquet manifest")
    declared_hash = manifest.get("manifest_content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_content_sha256", None)
    computed = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if declared_hash != computed:
        raise ValueError("Feed Parquet manifest content hash mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Feed Parquet manifest has no split records")
    for split in ("train", "validation"):
        declared = splits.get(split)
        if not isinstance(declared, Mapping):
            raise ValueError(f"Feed Parquet manifest misses {split}")
        actual = reports[split]
        if declared.get("sha256") != actual.get("sha256"):
            raise ValueError(f"Feed {split} Parquet hash differs from its manifest")
        if declared.get("output") != Path(str(actual.get("path"))).name:
            raise ValueError(f"Feed {split} Parquet filename differs from its manifest")
        if int(declared.get("rows", -1)) != int(actual.get("rows", -2)):
            raise ValueError(f"Feed {split} Parquet row count differs from its manifest")


def _validate_dataset_binding(
    dataset_dir: Path,
    *,
    identities: Mapping[str, set[tuple[str, str]]],
    reports: Mapping[str, Mapping[str, Any]],
    data_manifest: Path | None,
) -> None:
    from shopping_grpo.feed.manifest import verify_manifest
    from shopping_grpo.feed.schema import iter_jsonl

    if data_manifest is None:
        raise ValueError("dataset binding requires the Feed Parquet manifest")
    root = dataset_dir.expanduser().resolve()
    dataset_manifest = verify_manifest(root)
    parquet_manifest = json.loads(
        data_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    if (
        parquet_manifest.get("dataset_manifest_content_sha256")
        != dataset_manifest.get("manifest_content_sha256")
    ):
        raise ValueError("Feed Parquet manifest is bound to a different dataset")
    expected: dict[str, set[tuple[str, str]]] = {}
    source_task_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        rows = root / "online_rl_tasks" / f"{split}.jsonl"
        declared = dataset_manifest.get("files", {}).get(
            f"online_rl_tasks/{split}.jsonl"
        )
        if not isinstance(declared, Mapping) or declared.get("sha256") != _sha256(rows):
            raise ValueError(f"dataset manifest does not bind online_rl_tasks/{split}.jsonl")
        source_rows = list(iter_jsonl(rows))
        source_task_hashes[split] = _feed_task_content_sha256(source_rows)
        split_ids: set[tuple[str, str]] = set()
        for row in source_rows:
            seed = row.get("episode_seed") or {}
            persona = seed.get("persona") if isinstance(seed, Mapping) else {}
            split_ids.add(
                (
                    str(row.get("episode_id") or row.get("task_id") or ""),
                    str(persona.get("persona_id") if isinstance(persona, Mapping) else ""),
                )
            )
        expected[split] = split_ids
        parquet_split = parquet_manifest.get("splits", {}).get(split, {})
        if parquet_split.get("source_sha256") != _sha256(rows):
            raise ValueError(f"Feed Parquet manifest source hash mismatch for {split}")
    if identities["train"] != expected["train"]:
        raise ValueError("Feed train Parquet identities differ from the dataset")
    if identities["validation"] != expected["validation"]:
        raise ValueError("Feed validation Parquet identities differ from the dataset")
    for split in ("train", "validation"):
        if reports[split].get("task_content_sha256") != source_task_hashes[split]:
            raise ValueError(f"Feed {split} Parquet task content differs from the dataset")
    frozen = expected["test"]
    for split in ("train", "validation"):
        episodes = {item[0] for item in identities[split]}
        personas = {item[1] for item in identities[split]}
        if episodes & {item[0] for item in frozen}:
            raise ValueError(f"Feed {split} episodes overlap frozen test")
        if personas & {item[1] for item in frozen}:
            raise ValueError(f"Feed {split} personas overlap frozen test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SCHEMA_CONSTRAINT_KEYS = frozenset(
    {
        "type",
        "required",
        "additionalProperties",
        "enum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def validate_tool_schema_roundtrip(schema_class: Any) -> dict[str, Any]:
    """Prove veRL's schema model preserves the strict JSON-Schema contract.

    Some serving/schema models accept a tool declaration but silently discard
    constraints such as ``additionalProperties: false`` or ``maxItems``.  That
    is unsafe for this profile because the runtime guard and model-visible
    schema must describe the same action space.
    """

    tools_module = import_module("shopping_grpo.feed.tools")
    schemas = getattr(tools_module, "FEED_TOOL_SCHEMAS", None)
    if not isinstance(schemas, list):
        raise ValueError("Feed canonical tool schemas are unavailable")
    keyword_count = 0
    checked: list[str] = []
    for source_schema in schemas:
        try:
            if hasattr(schema_class, "model_validate"):
                runtime_schema = schema_class.model_validate(source_schema)
            elif hasattr(schema_class, "parse_obj"):
                runtime_schema = schema_class.parse_obj(source_schema)
            else:
                runtime_schema = schema_class(**source_schema)
            if hasattr(runtime_schema, "model_dump"):
                try:
                    dumped = runtime_schema.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                except TypeError:
                    dumped = runtime_schema.model_dump(by_alias=True, exclude_none=True)
            elif hasattr(runtime_schema, "dict"):
                dumped = runtime_schema.dict(by_alias=True, exclude_none=True)
            elif isinstance(runtime_schema, Mapping):
                dumped = dict(runtime_schema)
            else:
                raise TypeError("schema object has no mapping serializer")
        except Exception as exc:
            raise ValueError(
                "veRL rejected a canonical Feed tool schema: "
                f"{exc.__class__.__name__}"
            ) from exc

        expected = _schema_contract_projection(source_schema)
        actual = _schema_contract_projection(dumped)
        name = str(source_schema.get("function", {}).get("name", "unknown"))
        if actual != expected:
            raise ValueError(
                "veRL tool-schema roundtrip dropped or changed strict constraints "
                f"for {name}"
            )
        checked.append(name)
        keyword_count += _count_constraint_leaves(expected)
    return {
        "supported": True,
        "tools_checked": checked,
        "constraint_count": keyword_count,
    }


def verify_dynamic_sampling_patch(trainer_module: Any) -> str:
    """Verify the imported veRL trainer source carries the pinned patch marker."""

    return verify_source_marker(
        trainer_module,
        marker=DYNAMIC_SAMPLING_PATCH_MARKER,
        label="veRL ray_trainer",
    )


def verify_source_marker(module: Any, *, marker: str, label: str) -> str:
    """Resolve an imported module's source and require an exact patch marker."""

    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise ValueError(f"{label} module has no inspectable source path")
    path = Path(raw_path).resolve()
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(source_from_cache(str(path))).resolve()
        except ValueError as exc:
            raise ValueError(f"cannot resolve {label} source") from exc
    if not path.is_file():
        raise ValueError(f"{label} source is missing: {path}")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot inspect {label} source: {path}") from exc
    if marker not in source:
        raise ValueError(
            f"installed {label} is missing required patch marker {marker}"
        )
    return str(path)


def _schema_contract_projection(value: Any, *, properties: bool = False) -> Any:
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if properties:
                projected[key] = _schema_contract_projection(item)
            elif key in _SCHEMA_CONSTRAINT_KEYS:
                projected[key] = _schema_contract_projection(item)
            elif key == "properties":
                projected[key] = _schema_contract_projection(item, properties=True)
            elif key in {"function", "parameters", "items"}:
                projected[key] = _schema_contract_projection(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [_schema_contract_projection(item) for item in value]
    if hasattr(value, "value"):
        return _schema_contract_projection(value.value)
    return value


def _count_constraint_leaves(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            (1 if key in _SCHEMA_CONSTRAINT_KEYS else 0)
            + _count_constraint_leaves(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_count_constraint_leaves(item) for item in value)
    return 0


def _expect_adapter_constant(adapter: Any, name: str, expected: str) -> None:
    actual = getattr(adapter, name, None)
    if actual != expected:
        raise ValueError(
            f"Feed adapter {name} must be {expected!r}, got {actual!r}"
        )


def _parse_scalar(raw: str) -> Any:
    lowered = raw.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if raw.startswith(("'", '"')) and raw.endswith(raw[0]):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _omegaconf_default(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{oc\.env:[^,}]+,([^}]+)\}", value)
    return _parse_scalar(match.group(1)) if match else value


def _expect(values: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = values.get(path)
    if actual != expected:
        raise ValueError(f"{path} must be {expected!r}, got {actual!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="fail unless the pinned veRL 0.8 runtime is installed",
    )
    parser.add_argument(
        "--grpo-config",
        type=Path,
        help="validate this exact Feed GRPO config instead of the repository default",
    )
    parser.add_argument("--train-data", type=Path, help="exact training Parquet")
    parser.add_argument("--val-data", type=Path, help="exact validation Parquet")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        help="hash manifest produced beside the Feed Parquet files",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="source five-artifact dataset whose frozen test split must stay isolated",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)
    try:
        report = static_preflight(
            args.root,
            require_runtime=args.require_runtime,
            grpo_config=args.grpo_config,
            train_data=args.train_data,
            val_data=args.val_data,
            data_manifest=args.data_manifest,
            dataset_dir=args.dataset_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"Feed GRPO static preflight failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print("Feed GRPO static preflight passed (training was not started).")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
