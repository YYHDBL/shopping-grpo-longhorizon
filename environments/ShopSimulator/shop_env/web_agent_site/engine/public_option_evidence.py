"""Opt-in reviewed public-image evidence. Never inspect goals or target answers.

Manifest entries are bound to the catalog's exact ASIN/axis/value/image URL.
No remote fetch or model call occurs in the environment.
"""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path


def _norm(value):
    return re.sub(r"\s+", "", str(value)).casefold()


@lru_cache(maxsize=4)
def _load(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("version") != "public-option-evidence-v1":
        raise ValueError("Unrecognized public evidence version")
    records = data.get("records", [])
    seen = set()
    for record in records:
        key = (record["asin"], record["axis"], record["option_value"])
        if key in seen:
            raise ValueError("Duplicate public evidence record")
        seen.add(key)
        if (
            not record["source_url"].startswith("https://")
            or not record["caption"]
            or not isinstance(record["facts"], dict)
        ):
            raise ValueError("Invalid public evidence record")
    return records, hashlib.sha256(raw).hexdigest()


def option_evidence(product):
    path = os.environ.get("SHOP_PUBLIC_OPTION_EVIDENCE")
    if not path:
        return []
    records, digest = _load(path)
    result = []
    for r in records:
        if str(product.get("asin")) != r["asin"]:
            continue
        matches = [
            v
            for axis, vs in (product.get("customization_options") or {}).items()
            if _norm(axis) == _norm(r["axis"])
            for v in vs
            if _norm(v.get("value")) == _norm(r["option_value"])
        ]
        if len(matches) != 1 or matches[0].get("image") != r["source_url"]:
            raise ValueError("Public evidence binding does not match catalog option image")
        result.append({**r, "manifest_sha256": digest})
    return result


def selected_option_evidence(product, selected):
    return [
        r
        for r in option_evidence(product)
        if any(
            _norm(k) == _norm(r["axis"]) and _norm(v) == _norm(r["option_value"])
            for k, v in selected.items()
        )
    ]
