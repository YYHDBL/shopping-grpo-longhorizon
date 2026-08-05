"""Canonical tools exposed by the long-horizon feed-shopping profile.

The seven information tools gather public evidence.  ``commit_recommendation``
is the only policy action: its flat payload still has an explicit
When--What--How decomposition so trajectories remain easy to validate and
train on.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


MAX_INFO_TOOL_CALLS_PER_VIDEO = 3
COMMIT_TOOL_NAME = "commit_recommendation"
INFO_TOOL_NAMES = frozenset(
    {
        "retrieve_products",
        "inspect_product",
        "compare_products",
        "read_reviews",
        "find_alternatives",
        "find_complements",
        "check_inventory",
    }
)


class FeedToolValidationError(ValueError):
    """A model tool call does not conform to the public feed-tool schema."""


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _string(description, *, maximum=512):
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": maximum,
        "description": description,
    }


def _product_ids(description, *, minimum=1, maximum=4):
    return {
        "type": "array",
        "items": _string(
            "A product ID copied from visible_product_ids.", maximum=128
        ),
        "minItems": minimum,
        "maxItems": maximum,
        "uniqueItems": True,
        "description": description,
    }


FEED_TOOL_SCHEMAS = [
    _schema(
        "retrieve_products",
        "Retrieve products relevant to the current video and public behavior history. "
        "This is an information call and counts toward the limit of three calls per video.",
        {"query": _string("A concise category-and-constraint retrieval query.", maximum=500)},
        ["query"],
    ),
    _schema(
        "inspect_product",
        "Inspect public catalog facts for one currently visible product. This information "
        "call counts toward the limit of three calls per video.",
        {"product_id": _string("A product ID copied from visible_product_ids.", maximum=128)},
        ["product_id"],
    ),
    _schema(
        "compare_products",
        "Compare two to four currently visible products on public price, attributes, and "
        "fit. This information call counts toward the limit of three calls per video.",
        {"product_ids": _product_ids("Products to compare.", minimum=2)},
        ["product_ids"],
    ),
    _schema(
        "read_reviews",
        "Read the public review summary and disclosed risks for one currently visible "
        "product. This information call counts toward the limit of three calls per video.",
        {"product_id": _string("A product ID copied from visible_product_ids.", maximum=128)},
        ["product_id"],
    ),
    _schema(
        "find_alternatives",
        "Find cheaper or otherwise substitutable products for one currently visible product. "
        "This information call counts toward the limit of three calls per video.",
        {"product_id": _string("The visible product for which alternatives are requested.", maximum=128)},
        ["product_id"],
    ),
    _schema(
        "find_complements",
        "Find genuinely complementary products for one currently visible product. This "
        "information call counts toward the limit of three calls per video.",
        {"product_id": _string("The visible anchor product.", maximum=128)},
        ["product_id"],
    ),
    _schema(
        "check_inventory",
        "Check current public availability for one to four visible products. This information "
        "call counts toward the limit of three calls per video.",
        {"product_ids": _product_ids("Visible products whose inventory should be checked.")},
        ["product_ids"],
    ),
    _schema(
        COMMIT_TOOL_NAME,
        "Commit exactly one intervention for the current video. The flat fields implement "
        "When--What--How: decision is When; product_ids and relationship are What; surface "
        "and strategy are How. Evidence IDs must be copied from the current evidence_ids. "
        "Use no_recommend when intervention would be irrelevant or repetitive after purchase.",
        {
            "decision": {
                "type": "string",
                "enum": ["recommend", "delay", "no_recommend"],
                "description": "When: intervene now, delay, or do not recommend.",
            },
            "surface": {
                "type": "string",
                "enum": [
                    "none",
                    "product_card",
                    "coupon",
                    "review_summary",
                    "price_comparison",
                    "similar_products",
                    "bundle",
                    "creator_video",
                ],
                "description": "How: the user-visible intervention surface; use none otherwise.",
            },
            "product_ids": _product_ids(
                "What: visible products selected for the intervention; empty when not "
                "recommending.",
                minimum=0,
                maximum=2,
            ),
            "relationship": {
                "type": "string",
                "enum": ["primary", "alternative", "complement", "bundle"],
                "description": "What: role of the selected product set; use primary when not recommending.",
            },
            "strategy": {
                "type": "string",
                "enum": [
                    "none",
                    "direct",
                    "review_summary",
                    "price_comparison",
                    "discount",
                    "cheaper_alternative",
                    "similar",
                    "complement",
                    "bundle",
                ],
                "description": "How: a non-none presentation strategy for recommendations; use none otherwise.",
            },
            "evidence_ids": {
                "type": "array",
                "items": _string("An evidence ID copied from the current evidence_ids."),
                "minItems": 0,
                "maxItems": 16,
                "uniqueItems": True,
                "description": "Public evidence supporting this decision.",
            },
            "explanation": {
                "type": "string",
                "maxLength": 500,
                "description": "Optional concise, evidence-grounded user-facing explanation.",
            },
        },
        ["decision", "surface", "product_ids", "strategy", "evidence_ids"],
    ),
]

FEED_TOOL_SCHEMAS_BY_NAME = {
    schema["function"]["name"]: schema for schema in FEED_TOOL_SCHEMAS
}
FEED_TOOL_NAMES = tuple(FEED_TOOL_SCHEMAS_BY_NAME)
TOOL_ARGUMENT_NAMES = {
    name: frozenset(schema["function"]["parameters"]["properties"])
    for name, schema in FEED_TOOL_SCHEMAS_BY_NAME.items()
}


def schema_extra_argument_names(name: str, arguments: object) -> list[str]:
    """Return undeclared top-level argument names in deterministic order."""
    schema = FEED_TOOL_SCHEMAS_BY_NAME.get(name)
    if schema is None or not isinstance(arguments, Mapping):
        return []
    allowed = schema["function"]["parameters"]["properties"]
    return sorted(str(key) for key in arguments if key not in allowed)


def validate_tool_arguments(name: str, arguments: object) -> dict[str, Any]:
    """Validate and copy one call using only the JSON-Schema subset we publish.

    Keeping this small validator in the standard library makes the same strict
    contract available before an optional serving stack is installed.
    """
    schema = FEED_TOOL_SCHEMAS_BY_NAME.get(name)
    if schema is None:
        raise FeedToolValidationError(f"unknown_tool:{name}")
    if not isinstance(arguments, Mapping):
        raise FeedToolValidationError("arguments_must_be_object")

    parameters = schema["function"]["parameters"]
    extras = schema_extra_argument_names(name, arguments)
    if extras:
        raise FeedToolValidationError("schema_extra_arguments:" + ",".join(extras))
    missing = [field for field in parameters["required"] if field not in arguments]
    if missing:
        raise FeedToolValidationError("schema_missing_arguments:" + ",".join(missing))

    for field, value in arguments.items():
        _validate_value(value, parameters["properties"][field], field)
    return deepcopy(dict(arguments))


def _validate_value(value: object, schema: Mapping, path: str) -> None:
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            raise FeedToolValidationError(f"schema_type_error:{path}:string")
        if len(value) < int(schema.get("minLength", 0)):
            raise FeedToolValidationError(f"schema_min_length:{path}")
        if int(schema.get("minLength", 0)) > 0 and not value.strip():
            raise FeedToolValidationError(f"schema_blank_string:{path}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise FeedToolValidationError(f"schema_max_length:{path}")
    elif expected == "array":
        if not isinstance(value, list):
            raise FeedToolValidationError(f"schema_type_error:{path}:array")
        if len(value) < int(schema.get("minItems", 0)):
            raise FeedToolValidationError(f"schema_min_items:{path}")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise FeedToolValidationError(f"schema_max_items:{path}")
        if schema.get("uniqueItems") and len({_hashable(item) for item in value}) != len(value):
            raise FeedToolValidationError(f"schema_unique_items:{path}")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_value(item, item_schema, f"{path}[{index}]")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise FeedToolValidationError(f"schema_type_error:{path}:integer")
    elif expected == "object":
        if not isinstance(value, Mapping):
            raise FeedToolValidationError(f"schema_type_error:{path}:object")
    elif expected is not None:
        raise FeedToolValidationError(f"unsupported_schema_type:{path}:{expected}")

    if "enum" in schema and value not in schema["enum"]:
        raise FeedToolValidationError(f"schema_enum_error:{path}")


def _hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    return value


def feed_tool_call_to_action(name: str, parameters: object):
    """Normalize a tool call into a JSON-safe runtime envelope or action.

    Information calls remain an envelope because they do not advance the feed.
    A commit is adapted through ``FeedAction.from_dict`` when the typed contract
    is installed.  The lazy fallback keeps this module independently importable
    while the parallel simulator profile is being assembled.
    """
    normalized = validate_tool_arguments(name, parameters)
    if name != COMMIT_TOOL_NAME:
        return {"tool": name, "arguments": normalized}
    try:
        from shopping_grpo.feed.schema import FeedAction
    except (ImportError, AttributeError):
        return normalized
    return FeedAction.from_dict(normalized)


# Match the established ShopSimulator profile's public helper name.
tool_call_to_action = feed_tool_call_to_action


__all__ = [
    "COMMIT_TOOL_NAME",
    "FEED_TOOL_NAMES",
    "FEED_TOOL_SCHEMAS",
    "FEED_TOOL_SCHEMAS_BY_NAME",
    "FeedToolValidationError",
    "INFO_TOOL_NAMES",
    "MAX_INFO_TOOL_CALLS_PER_VIDEO",
    "TOOL_ARGUMENT_NAMES",
    "feed_tool_call_to_action",
    "schema_extra_argument_names",
    "tool_call_to_action",
    "validate_tool_arguments",
]
