"""Component B: public option completeness, known price and explicit budget checks."""

import json
import re
from decimal import Decimal

from shopping_grpo.environment.budget import check_purchase_budget


def field(observation, name, default=None):
    match = re.search(r"^" + re.escape(name) + r":[ \t]*([^\n]*)$", observation, re.MULTILINE)
    if not match:
        return default
    try:
        return json.loads(match[1])
    except json.JSONDecodeError:
        return default


def purchase_check(instruction, observation):
    # Missing or malformed JSON is unknown, not an explicitly empty option map.
    available = field(observation, "available_options")
    selected = field(observation, "selected_options")
    if (
        not isinstance(available, dict)
        or not isinstance(selected, dict)
        or any(
            not axis.strip()
            or not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            for axis, values in available.items()
        )
        or any(
            axis not in available or not isinstance(value, str) or value not in available[axis]
            for axis, value in selected.items()
        )
    ):
        return {
            "reason": "variant_state_unreadable",
            "feedback": "无法读取当前规格状态，购买未执行。请重新打开商品核验。",
        }
    missing = {
        axis: values for axis, values in available.items() if values and axis not in selected
    }
    if missing:
        return {
            "reason": "variant_axes_unselected",
            "feedback": "购买未执行：这些公开规格轴尚未选择（单值也需显式选择）："
            + json.dumps(missing, ensure_ascii=False)
            + "。请在当前商品页补齐再核对价格。",
        }
    if not re.search(r"^price:[ \t]*\d+(?:\.\d+)?[ \t]*$", observation, re.MULTILINE):
        return {
            "reason": "variant_price_unknown",
            "feedback": "购买未执行：当前没有完整规格的单一确定价格，请核验规格和价格。",
        }
    result = check_purchase_budget(instruction, observation)
    if result["reason"]:
        return result
    # Cover additional explicit wording, never turn approximations into hard caps.
    bounds = []
    for clause in re.split(r"[，,。；;！？?\n]", instruction):
        if re.search(r"左右|上下|大概|大约|差不多|最好|约", clause):
            continue
        for pattern in [
            r"不超\s*(\d+(?:\.\d+)?)\s*(?:元|块)",
            r"(?:预算|价格)\s*(?:控制在|有|为|是)\s*(\d+(?:\.\d+)?)\s*(?:元|块)",
            r"(\d+(?:\.\d+)?)\s*(?:元|块)?能搞定",
        ]:
            bounds += [Decimal(m[1]) for m in re.finditer(pattern, clause)]
    price = Decimal(re.search(r"^price:[ \t]*(\d+(?:\.\d+)?)", observation, re.MULTILINE)[1])
    if bounds and price > min(bounds):
        return {
            "reason": "budget_above_explicit_amount",
            "feedback": f"购买未执行：当前{price}元超过公开明确预算{min(bounds)}元，请换规格或候选，不要重复购买。",
        }
    return {"reason": None, "feedback": None}
