"""Fail-closed rendering for the model-visible feed observation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from shopping_grpo.feed.tools import MAX_INFO_TOOL_CALLS_PER_VIDEO


FEED_OBSERVATION_VERSION = "feed-observation-v1"
FEED_OBSERVATION_HEADER = "[FEED_OBSERVATION_V1]"
PUBLIC_OBSERVATION_FIELDS = (
    "observation_version",
    "environment_version",
    "episode_id",
    "step",
    "total_steps",
    "persona",
    "current_video",
    "recent_events",
    "cart",
    "purchased_product_ids",
    "visible_product_ids",
    "evidence_ids",
    "info_tool_calls",
    "max_info_tool_calls",
    "done",
)

# A safe top-level key must still not hide a latent value in a nested mapping.
# These fragments are deliberately checked recursively and case-insensitively.
SENSITIVE_KEY_FRAGMENTS = (
    "latent",
    "probability",
    "_prob",
    "prob_",
    "propensity",
    "likelihood",
    "true_",
    "reward",
    "hidden",
    "intent",
    "trust",
    "fatigue",
    "budget_remaining",
    "short_term_interest",
    "price_sensitivity",
    "satisfaction",
    "readiness",
    "hindsight",
    "qualified",
    "hard_match",
    "soft_satisfaction",
    "oracle",
    "gold_",
    "target_",
)


class FeedObservationError(ValueError):
    """The environment supplied malformed or non-public feed state."""


def render_feed_observation(state: Mapping) -> str:
    """Render only the frozen public allowlist and reject possible leakage.

    Unknown fields are rejected rather than silently dropped so a simulator
    change cannot accidentally train against a latent-state side channel.
    """
    if not isinstance(state, Mapping):
        raise FeedObservationError("observation_state must be an object")
    if state.get("observation_version") != FEED_OBSERVATION_VERSION:
        raise FeedObservationError("unsupported observation_state version")

    unknown = sorted(str(key) for key in state if key not in PUBLIC_OBSERVATION_FIELDS)
    if unknown:
        raise FeedObservationError(
            "observation_state contains non-public fields: " + ", ".join(unknown)
        )
    _reject_sensitive_keys(state)
    public_state = {
        field: _copy_json_value(state[field], field)
        for field in PUBLIC_OBSERVATION_FIELDS
        if field in state
    }
    _validate_public_control_fields(public_state)

    try:
        payload = json.dumps(
            public_state,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FeedObservationError(f"observation_state is not JSON-safe: {exc}") from exc
    return FEED_OBSERVATION_HEADER + "\n" + payload


def parse_feed_observation(rendered: str) -> dict[str, Any]:
    """Parse this renderer's public payload; never use it as action-guard state."""
    prefix = FEED_OBSERVATION_HEADER + "\n"
    if not isinstance(rendered, str) or not rendered.startswith(prefix):
        raise FeedObservationError("missing feed observation header")
    try:
        payload = json.loads(rendered[len(prefix) :])
    except json.JSONDecodeError as exc:
        raise FeedObservationError("invalid feed observation JSON") from exc
    if not isinstance(payload, dict):
        raise FeedObservationError("feed observation payload must be an object")
    return payload


def _reject_sensitive_keys(value: object, path: str = "observation_state") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            folded = key.casefold()
            if any(fragment in folded for fragment in SENSITIVE_KEY_FRAGMENTS):
                raise FeedObservationError(
                    f"observation_state contains sensitive field: {path}.{key}"
                )
            _reject_sensitive_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_keys(item, f"{path}[{index}]")


def _copy_json_value(value: object, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeedObservationError(f"non-finite number at {path}")
        return value
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        copied = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise FeedObservationError(f"non-string object key at {path}")
            copied[raw_key] = _copy_json_value(item, f"{path}.{raw_key}")
        return copied
    raise FeedObservationError(f"non-JSON value at {path}: {type(value).__name__}")


def _validate_public_control_fields(state: Mapping) -> None:
    for field in ("visible_product_ids", "evidence_ids", "purchased_product_ids"):
        if field in state and not _is_string_list(state[field]):
            raise FeedObservationError(f"{field} must be a list of non-empty strings")

    calls = state.get("info_tool_calls")
    if calls is not None:
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise FeedObservationError("info_tool_calls must be a non-negative integer")
        if calls > MAX_INFO_TOOL_CALLS_PER_VIDEO:
            raise FeedObservationError("info_tool_calls exceeds the per-video limit")
    maximum = state.get("max_info_tool_calls")
    if maximum is not None:
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < 0
            or maximum > MAX_INFO_TOOL_CALLS_PER_VIDEO
        ):
            raise FeedObservationError(
                "max_info_tool_calls must be an integer between zero and "
                f"{MAX_INFO_TOOL_CALLS_PER_VIDEO}"
            )
        if calls is not None and calls > maximum:
            raise FeedObservationError("info_tool_calls exceeds max_info_tool_calls")


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


# Friendly aliases for callers that use the existing profile's naming convention.
OBSERVATION_VERSION = FEED_OBSERVATION_VERSION
HEADER = FEED_OBSERVATION_HEADER
StructuredObservationError = FeedObservationError
render_structured_observation = render_feed_observation


__all__ = [
    "FEED_OBSERVATION_HEADER",
    "FEED_OBSERVATION_VERSION",
    "FeedObservationError",
    "HEADER",
    "OBSERVATION_VERSION",
    "PUBLIC_OBSERVATION_FIELDS",
    "SENSITIVE_KEY_FRAGMENTS",
    "StructuredObservationError",
    "parse_feed_observation",
    "render_feed_observation",
    "render_structured_observation",
]
