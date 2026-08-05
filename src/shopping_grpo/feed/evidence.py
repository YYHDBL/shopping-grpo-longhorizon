"""Stable, bounded identifiers for public Feed evidence.

Evidence IDs are copied verbatim by the model, so they must be short, deterministic,
and safe for dot-delimited matching even when a catalog uses IDs such as ``P.1``.
"""

from __future__ import annotations

import hashlib
from urllib.parse import quote


MAX_EVIDENCE_COMPONENT_CHARS = 96


def evidence_component(value: object) -> str:
    """Encode one value as a bounded component that never contains ``.``."""

    raw = str(value)
    encoded = quote(raw, safe="-_~").replace(".", "%2E")
    if len(encoded) <= MAX_EVIDENCE_COMPONENT_CHARS:
        return encoded
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    prefix = encoded[: MAX_EVIDENCE_COMPONENT_CHARS - len(digest) - 1]
    return f"{prefix}~{digest}"


def has_product_evidence(evidence_ids: list[str] | tuple[str, ...], product_id: str) -> bool:
    """Return whether a canonical evidence ID explicitly owns ``product_id``."""

    token = evidence_component(product_id)
    return any(token in str(identifier).split(".") for identifier in evidence_ids)


__all__ = [
    "MAX_EVIDENCE_COMPONENT_CHARS",
    "evidence_component",
    "has_product_evidence",
]
