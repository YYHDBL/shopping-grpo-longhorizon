"""Machine-readable action guard for the feed-shopping tool contract."""

from __future__ import annotations

import json
from collections.abc import Mapping

from shopping_grpo.feed.evidence import has_product_evidence
from shopping_grpo.feed.tools import (
    COMMIT_TOOL_NAME,
    INFO_TOOL_NAMES,
    MAX_INFO_TOOL_CALLS_PER_VIDEO,
    FeedToolValidationError,
    feed_tool_call_to_action,
    schema_extra_argument_names,
    validate_tool_arguments,
)


RUNTIME_GUARD_FIELD = "runtime_action_guard"
GUARD_STATE_FIELDS = (
    "visible_product_ids",
    "evidence_ids",
    "purchased_product_ids",
    "info_tool_calls",
)


def action_reject_reason(name: str, arguments: object, state: object) -> str | None:
    """Return a stable rejection code, or ``None`` when a call is safe.

    ``state`` must be a mapping supplied directly by the environment.  This
    guard never extracts product or evidence IDs from rendered model text.
    """
    extras = schema_extra_argument_names(name, arguments)
    if extras:
        return "schema_extra_arguments:" + ",".join(extras)
    try:
        normalized = validate_tool_arguments(name, arguments)
    except FeedToolValidationError as exc:
        return str(exc)

    state_error = _guard_state_error(state)
    if state_error:
        return state_error
    assert isinstance(state, Mapping)  # established by _guard_state_error

    visible = set(state["visible_product_ids"])
    evidence = set(state["evidence_ids"])
    purchased = set(state["purchased_product_ids"])
    info_calls = state["info_tool_calls"]

    maximum = int(state.get("max_info_tool_calls", MAX_INFO_TOOL_CALLS_PER_VIDEO))
    if name in INFO_TOOL_NAMES and info_calls >= maximum:
        return "max_info_tool_calls_exceeded"

    for product_id in _referenced_product_ids(name, normalized):
        if product_id not in visible:
            return f"product_not_visible:{product_id}"

    if name != COMMIT_TOOL_NAME:
        return None

    for evidence_id in normalized["evidence_ids"]:
        if evidence_id not in evidence:
            return f"evidence_not_visible:{evidence_id}"

    decision = normalized["decision"]
    product_ids = normalized["product_ids"]
    surface = normalized["surface"]
    if decision == "recommend":
        if purchased.intersection(product_ids):
            return "repeat_marketing_after_purchase"
        if not product_ids:
            return "recommendation_requires_product"
        if not normalized["evidence_ids"]:
            return "recommendation_requires_evidence"
        if surface == "none":
            return "recommendation_requires_surface"
        if normalized["strategy"] == "none":
            return "recommendation_requires_strategy"
        relationship = normalized.get("relationship", "primary")
        if len(product_ids) == 2 and relationship != "bundle":
            return "multi_product_requires_bundle_relationship"
        if relationship == "bundle" and len(product_ids) != 2:
            return "bundle_relationship_requires_two_products"
        if normalized["strategy"] == "bundle" and len(product_ids) != 2:
            return "bundle_strategy_requires_two_products"
        if surface == "bundle" and len(product_ids) != 2:
            return "bundle_surface_requires_two_products"
        for product_id in product_ids:
            if not has_product_evidence(normalized["evidence_ids"], product_id):
                return f"missing_product_evidence:{product_id}"
        if not any(
            evidence_id.startswith(("video.", "history.", "persona."))
            for evidence_id in normalized["evidence_ids"]
        ):
            return "missing_context_evidence"
    else:
        if product_ids:
            return "non_recommendation_has_products"
        if surface != "none":
            return "non_recommendation_has_surface"
        if normalized["strategy"] != "none":
            return "non_recommendation_has_strategy"
        if normalized.get("relationship", "primary") != "primary":
            return "non_recommendation_has_relationship"
    return None


def _guard_state_error(state: object) -> str | None:
    if not isinstance(state, Mapping):
        return "invalid_guard_state:not_an_object"
    missing = [field for field in GUARD_STATE_FIELDS if field not in state]
    if missing:
        return "invalid_guard_state:missing_" + ",".join(missing)
    for field in GUARD_STATE_FIELDS[:3]:
        value = state[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            return f"invalid_guard_state:{field}"
    calls = state["info_tool_calls"]
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        return "invalid_guard_state:info_tool_calls"
    if "max_info_tool_calls" in state:
        maximum = state["max_info_tool_calls"]
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < 0
            or maximum > MAX_INFO_TOOL_CALLS_PER_VIDEO
        ):
            return "invalid_guard_state:max_info_tool_calls"
        if calls > maximum:
            return "invalid_guard_state:info_tool_calls"
    return None


def _referenced_product_ids(name: str, arguments: Mapping) -> list[str]:
    if name in {"inspect_product", "read_reviews", "find_alternatives", "find_complements"}:
        return [arguments["product_id"]]
    if name in {"compare_products", "check_inventory", COMMIT_TOOL_NAME}:
        return list(arguments["product_ids"])
    return []


def guarded_tool_call(name: str, arguments: object, state: object):
    """Validate a call against state, then convert it to the runtime action."""
    reason = action_reject_reason(name, arguments, state)
    if reason is not None:
        raise FeedToolValidationError(reason)
    return feed_tool_call_to_action(name, arguments)


def action_guard_tool_message(tool_call: Mapping, reason: str, state: object) -> dict:
    """Build a structured tool error without embedding the whole observation."""
    public_guard_state = _public_guard_state(state)
    code, _, detail = str(reason).partition(":")
    payload = {
        "ok": False,
        "error": {"code": code, "detail": detail or None},
        "guard_state": public_guard_state,
    }
    function = tool_call.get("function") if isinstance(tool_call, Mapping) else None
    function = function if isinstance(function, Mapping) else {}
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id") if isinstance(tool_call, Mapping) else None,
        "name": function.get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }


def _public_guard_state(state: object) -> dict:
    if not isinstance(state, Mapping):
        return {}
    result = {}
    for field in GUARD_STATE_FIELDS:
        value = state.get(field)
        if isinstance(value, list):
            result[field] = [item for item in value if isinstance(item, str)]
        elif field == "info_tool_calls" and isinstance(value, int) and not isinstance(value, bool):
            result[field] = value
    maximum = state.get("max_info_tool_calls", MAX_INFO_TOOL_CALLS_PER_VIDEO)
    result["max_info_tool_calls"] = (
        maximum
        if isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and 0 <= maximum <= MAX_INFO_TOOL_CALLS_PER_VIDEO
        else MAX_INFO_TOOL_CALLS_PER_VIDEO
    )
    return result


# Explicit feed-prefixed aliases are convenient in code that imports both profiles.
feed_action_reject_reason = action_reject_reason
feed_action_guard_tool_message = action_guard_tool_message


__all__ = [
    "GUARD_STATE_FIELDS",
    "RUNTIME_GUARD_FIELD",
    "action_guard_tool_message",
    "action_reject_reason",
    "feed_action_guard_tool_message",
    "feed_action_reject_reason",
    "guarded_tool_call",
]
