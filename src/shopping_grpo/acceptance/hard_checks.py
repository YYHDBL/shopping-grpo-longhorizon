"""第一级验收：纯本地代码硬约束，零 API 成本。

只检查可编程确定的约束：结构完整性、必选规格存在且可选、预算上限。
品牌与类别一致性不做词面检查（中文品牌词提取误报率高），交给第二级 Jev
语义判断——它能看到 title / shop_name / category，天然覆盖品牌与品类。
价格判定复用环境 Reward 同款 ``resolve_variant_price``，保证验收与判分
使用同一把尺子。
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path
from typing import Mapping

HARD_CHECKS_VERSION = "acceptance-hard-v1"

# 环境引擎路径：验收需要与 Reward 完全一致的 variant 价格逻辑。
# 不把 ShopSimulator 目录加入 sys.path——那里也有一个 scripts 包，会遮蔽
# 仓库根目录的 scripts 包并破坏全量测试收集。改为受控注册包后再走正常
# 导入机制；web_agent_site 与其 engine 子包的 __init__ 均为空文件。
_ENV_ENGINE = Path(__file__).resolve().parents[3] / (
    "environments/ShopSimulator/shop_env"
)


def _load_variant_price():
    site = _ENV_ENGINE / "web_agent_site"
    if "web_agent_site" not in sys.modules:
        package = types.ModuleType("web_agent_site")
        package.__path__ = [str(site)]
        sys.modules["web_agent_site"] = package
    if "web_agent_site.engine" not in sys.modules:
        engine = types.ModuleType("web_agent_site.engine")
        engine.__path__ = [str(site / "engine")]
        sys.modules["web_agent_site.engine"] = engine
    return importlib.import_module("web_agent_site.engine.variant_price")


_variant_price = _load_variant_price()
VARIANT_PRICE_VERSION = _variant_price.VARIANT_PRICE_VERSION
resolve_variant_price = _variant_price.resolve_variant_price

PASS = "pass"
FAIL = "fail"
UNVERIFIABLE = "unverifiable"

# 中文预算表述：取所有命中的上限里最严格的一个。
_BUDGET_PATTERNS = [
    # 预算在 1000 元以下 / 预算 500 以内 / 预算不超过3000元
    re.compile(r"预算[^。,，;；]{0,6}?(\d+(?:\.\d+)?)\s*万?(?:元)?(?:以?内|以下|之下|不等|左右以内)"),
    # 1000 元以内 / 300元以下 / 不超过 500 元 / 低于200元
    re.compile(r"(?:不超过|低于|少于)\s*(\d+(?:\.\d+)?)\s*万?(?:元)?"),
    re.compile(r"(\d+(?:\.\d+)?)\s*万?元(?:以?内|之内|以下|之下)"),
]
_WAN_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*万")


def parse_budget_limit(instruction: str):
    """从 instruction 解析预算上限；返回 (limit, evidence) 或 (None, 原因)。"""
    text = str(instruction or "")
    limits = []
    for pattern in _BUDGET_PATTERNS:
        for match in pattern.finditer(text):
            value = float(match.group(1))
            # "2万" 这种单位在万位 pattern 里已由 "万?" 分组吞掉，这里显式换算。
            span = match.group(0)
            if _WAN_PATTERN.fullmatch(span.replace("元", "").strip()) or "万" in span:
                value *= 10000.0
            limits.append((value, match.group(0)))
    if not limits:
        return None, "budget_not_declared"
    limit, evidence = min(limits, key=lambda item: item[0])
    return limit, evidence


def _selected_options_from_instruction(record: Mapping):
    """把 instruction_options 的值定位到 customization_options 的规格轴。"""
    instructions = record.get("instructions") or []
    required = [str(v) for v in ((instructions[0] or {}).get("instruction_options") or [])]
    options = record.get("customization_options") or {}
    selected: dict[str, str] = {}
    unavailable: list[str] = []
    not_found: list[str] = []
    axis_conflicts: list[str] = []
    for value in required:
        matches = [
            (axis, entry)
            for axis, entries in options.items()
            for entry in entries
            if str(entry.get("value")) == value
        ]
        if not matches:
            not_found.append(f"{value} (not found in any option axis)")
            continue
        axes = {axis for axis, _ in matches}
        if len(axes) > 1:
            axis_conflicts.append(f"{value} appears on multiple axes: {sorted(axes)}")
            continue
        axis, entry = matches[0]
        if axis in selected and selected[axis] != value:
            axis_conflicts.append(f"{axis} required with two values: {selected[axis]}, {value}")
            continue
        if entry.get("is_available") is False:
            unavailable.append(f"{value} (is_available=false on {axis})")
            continue
        selected[axis] = value
    missing = unavailable + not_found
    return selected, missing, axis_conflicts, required


def _check_structure(record: Mapping) -> dict:
    problems = []
    if not str(record.get("asin") or "").strip():
        problems.append("missing asin")
    if not str(record.get("title") or "").strip():
        problems.append("missing title")
    if not str(record.get("category") or "").strip():
        problems.append("missing category")
    pricing = record.get("pricing")
    if not isinstance(pricing, list) or not pricing:
        problems.append("missing pricing")
    instructions = record.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        problems.append("missing instructions")
    else:
        instruction = str((instructions[0] or {}).get("instruction") or "")
        if not instruction.strip():
            problems.append("missing instruction text")
    return {
        "name": "structure",
        "status": FAIL if problems else PASS,
        "detail": "; ".join(problems) if problems else "all required fields present",
    }


def _build_required_check(selected, missing, axis_conflicts, required) -> dict:
    detail = {
        "required_count": len(required),
        "resolved_axes": sorted(selected),
    }
    if axis_conflicts:
        return {
            "name": "required_options",
            "status": FAIL,
            "detail": {**detail, "axis_conflicts": axis_conflicts},
        }
    if missing:
        return {
            "name": "required_options",
            "status": FAIL,
            "detail": {**detail, "missing": missing},
        }
    return {
        "name": "required_options",
        "status": PASS,
        "detail": {**detail, "selected": selected},
    }


def _check_budget(record: Mapping, selected: dict[str, str]) -> dict:
    instructions = record.get("instructions") or []
    instruction = str((instructions[0] or {}).get("instruction") or "")
    limit, evidence = parse_budget_limit(instruction)
    if limit is None:
        return {
            "name": "budget",
            "status": PASS,
            "detail": "budget_not_declared (same convention as reward price gate)",
        }
    product = {"customization_options": record.get("customization_options") or {}}
    resolution = resolve_variant_price(product, selected)
    check = {
        "name": "budget",
        "detail": {
            "limit": limit,
            "limit_evidence": evidence,
            "variant_price_version": VARIANT_PRICE_VERSION,
            "price_resolution": resolution,
        },
    }
    if resolution.get("status") != PASS:
        check["status"] = UNVERIFIABLE
        return check
    price = resolution.get("price")
    check["detail"]["price"] = price
    if price is None or price <= limit:
        check["status"] = PASS
    else:
        check["status"] = FAIL
        check["detail"]["exceeded_by"] = round(price - limit, 2)
    return check


def run_hard_checks(record: Mapping) -> dict:
    """对一条记录执行全部硬约束检查，返回逐项明细与汇总状态。"""
    structure = _check_structure(record)
    if structure["status"] == FAIL:
        return {
            "record_id": str(record.get("asin") or ""),
            "version": HARD_CHECKS_VERSION,
            "checks": [structure],
            "status": FAIL,
        }
    selected, missing, axis_conflicts, required = _selected_options_from_instruction(record)
    required_check = _build_required_check(selected, missing, axis_conflicts, required)
    budget = _check_budget(record, selected)
    checks = [structure, required_check, budget]
    statuses = [c["status"] for c in checks]
    if FAIL in statuses:
        status = FAIL
    elif UNVERIFIABLE in statuses:
        status = UNVERIFIABLE
    else:
        status = PASS
    return {
        "record_id": str(record.get("asin")),
        "version": HARD_CHECKS_VERSION,
        "checks": checks,
        "status": status,
    }
