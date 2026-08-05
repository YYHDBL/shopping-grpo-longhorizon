"""Frozen-test rollout driver for an OpenAI-compatible tool-calling model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from shopping_grpo.feed.actions import action_reject_reason
from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.manifest import canonical_json, verify_manifest
from shopping_grpo.feed.schema import EpisodeSeed, iter_jsonl, write_jsonl
from shopping_grpo.feed.simulator import FeedShoppingEnv
from shopping_grpo.feed.tools import FEED_TOOL_SCHEMAS
from shopping_grpo.feed.verl_adapter import build_public_tool_payload


Completion = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Mapping[str, Any]]


def _tool_call(message: Mapping[str, Any]) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("model must emit exactly one tool call")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        raise ValueError("model tool call has no function")
    call_id = str(call.get("id") or "")
    name = str(function.get("name") or "")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("model tool arguments are not JSON") from exc
    if not call_id or not name or not isinstance(arguments, dict):
        raise ValueError("model tool call is incomplete")
    assistant = {
        "role": "assistant",
        "content": str(message.get("content") or ""),
        "tool_calls": [deepcopy(dict(call))],
    }
    return assistant, call_id, name, arguments


def _error_reply(call_id: str, name: str, reason: str) -> dict[str, Any]:
    code, _, detail = str(reason).partition(":")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": canonical_json(
            {
                "payload_version": "feed-tool-delta-v1",
                "ok": False,
                "tool": name,
                "error": {"code": code, "detail": detail or None},
            }
        ),
    }


def rollout_task(
    task: Mapping[str, Any],
    complete: Completion,
    *,
    policy_id: str,
    max_rejections_per_step: int = 8,
) -> dict[str, Any]:
    """Run one self-contained test task while exposing only public messages."""

    seed = EpisodeSeed.from_dict(task["episode_seed"])
    if not 24 <= len(seed.videos) <= 48:
        raise ValueError("frozen model task must contain 24--48 videos")
    catalog = ProductCatalog(task["catalog_products"])
    env = FeedShoppingEnv(seed, catalog, calibration=task.get("calibration"))
    messages = deepcopy(list(task["prompt"]))
    tools = deepcopy(list(task.get("tools") or FEED_TOOL_SCHEMAS))
    if tools != FEED_TOOL_SCHEMAS:
        raise ValueError("frozen model task tool schemas differ from the canonical contract")
    transitions = []
    while not env.done:
        pre_observation = env.observation()
        rejections = 0
        while True:
            proposal = complete(deepcopy(messages), deepcopy(tools))
            assistant, call_id, name, arguments = _tool_call(proposal)
            messages.append(assistant)
            reason = action_reject_reason(name, arguments, env.observation())
            if reason is not None:
                rejections += 1
                messages.append(_error_reply(call_id, name, reason))
                if rejections > max_rejections_per_step:
                    raise RuntimeError("model exceeded the per-step action rejection budget")
                continue
            try:
                raw_result = env.call_tool(name, arguments)
                observation = env.observation()
                payload = build_public_tool_payload(
                    name,
                    raw_result,
                    observation,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"frozen rollout tool execution failed: {exc.__class__.__name__}"
                ) from exc
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": canonical_json(payload),
                }
            )
            if name == "commit_recommendation":
                transition = deepcopy(env.transitions[-1])
                transition["pre_observation"] = pre_observation
                transitions.append(transition)
                break
    return {
        "artifact_version": "feed-model-rollout-v1",
        "trajectory_id": f"{seed.episode_id}:{policy_id}",
        "episode_id": seed.episode_id,
        "persona_id": seed.persona.persona_id,
        "split": "test",
        "policy": policy_id,
        "transitions": transitions,
        "evaluator_summary": {**env.summary(), "evaluator_only": True},
    }


def rollout_frozen_dataset(
    dataset_dir: str | Path,
    output: str | Path,
    complete: Completion,
    *,
    policy_id: str,
) -> list[dict[str, Any]]:
    root = Path(dataset_dir).resolve()
    manifest = verify_manifest(root)
    tasks_path = root / "online_rl_tasks" / "test.jsonl"
    declared = manifest.get("files", {}).get("online_rl_tasks/test.jsonl")
    from shopping_grpo.feed.manifest import sha256_file

    if not isinstance(declared, Mapping) or declared.get("sha256") != sha256_file(tasks_path):
        raise ValueError("dataset manifest does not bind frozen online RL tasks")
    rows = [
        rollout_task(task, complete, policy_id=policy_id)
        for task in iter_jsonl(tasks_path)
    ]
    if not rows:
        raise ValueError("frozen test task set is empty")
    write_jsonl(output, rows)
    return rows


__all__ = ["rollout_frozen_dataset", "rollout_task"]
