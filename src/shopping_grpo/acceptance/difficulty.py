"""基线 4.1 确认的二维难度规则，验收报告与后续切分共用。"""

from __future__ import annotations

from math import prod
from typing import Mapping


def difficulty_of(record: Mapping) -> str:
    """attributes 数量 × 规格组合数的二维规则；标注缺失按最低档处理。"""
    instructions = record.get("instructions") or []
    attributes = (instructions[0] or {}).get("attributes") if instructions else None
    n_attributes = len(attributes or [])
    options = record.get("customization_options") or {}
    combinations = prod(len(values) for values in options.values()) if options else 1
    if n_attributes < 4 and combinations <= 7:
        return "easy"
    if n_attributes >= 9 or combinations >= 21:
        return "hard"
    return "medium"
