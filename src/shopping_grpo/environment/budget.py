"""Conservative purchase-time budget guard using only public user/page text."""

import re
from decimal import Decimal

VERSION = "explicit-cny-budget-v1"
N = r"(\d+(?:\.\d+)?)"
CURRENCY = r"(?:元|块钱|块)"
UPPER_PATTERNS = (
    rf"(?:不超过|不得超过|不能超过|不高于|最多|至多|上限(?:为|是)?)[ ]*{N}[ ]*{CURRENCY}",
    rf"{N}[ ]*{CURRENCY}[ ]*(?:以内|以下)",
)
LOWER_PATTERNS = (
    rf"(?:不少于|不低于|至少)[ ]*{N}[ ]*{CURRENCY}",
    rf"{N}[ ]*{CURRENCY}[ ]*(?:以上)",
)
RANGE_PATTERN = rf"(?:价格|价位|预算)(?:要|需要)?(?:在|为|是)?[ ]*{N}[ ]*(?:元)?[ ]*[-—–~～至到][ ]*{N}[ ]*{CURRENCY}(?:之间)?"


def extract_budget(instruction):
    """Approximate, vague and unsupported wording remains explicitly uncovered."""
    lower, upper, evidence = [], [], []
    for clause in re.split(r"[，,。；;！!？?\n]", instruction):
        if re.search(r"左右|上下|大概|大约|差不多|最好|尽量|约莫", clause):
            continue
        for match in re.finditer(RANGE_PATTERN, clause):
            lower.append(Decimal(match[1]))
            upper.append(Decimal(match[2]))
            evidence.append(match[0])
        for patterns, bucket in ((UPPER_PATTERNS, upper), (LOWER_PATTERNS, lower)):
            for pattern in patterns:
                for match in re.finditer(pattern, clause):
                    bucket.append(Decimal(match[1]))
                    evidence.append(match[0])
    return {
        "lower": str(max(lower)) if lower else None,
        "upper": str(min(upper)) if upper else None,
        "evidence": list(dict.fromkeys(evidence)),
        "covered": bool(lower or upper),
        "version": VERSION,
    }


def check_purchase_budget(instruction, observation):
    bounds = extract_budget(instruction)
    result = {"bounds": bounds, "price": None, "reason": None, "feedback": None}
    if not bounds["covered"]:
        result["status"] = "uncovered"
        return result
    match = re.search(r"(?m)^price:[ \t]*(\d+(?:\.\d+)?)[ \t]*$", observation)
    price = Decimal(match[1]) if match else None
    result["price"] = str(price) if price is not None else None
    lower = Decimal(bounds["lower"]) if bounds["lower"] is not None else None
    upper = Decimal(bounds["upper"]) if bounds["upper"] is not None else None
    if lower is not None and upper is not None and lower > upper:
        reason, description = "conflicting_bounds", "用户的明确预算条件互相冲突，无法核验通过"
    elif price is None:
        reason, description = (
            "price_unverified",
            "当前页面未给出所选规格的单一确定价格，不能确认预算符合要求",
        )
    elif lower is not None and price < lower:
        reason, description = "below_minimum", f"当前价格 {price} 元低于用户明确下限 {lower} 元"
    elif upper is not None and price > upper:
        reason, description = "above_maximum", f"当前价格 {price} 元超过用户明确上限 {upper} 元"
    else:
        result["status"] = "allowed"
        return result
    result.update(
        status="blocked",
        reason="budget_" + reason,
        feedback=f"购买前预算检查拒绝，购买未执行：{description}。"
        "请依据当前页面修正规格并核实确定价格，或调用 back_to_search 返回后寻找其他商品。"
        "不要重复购买同一不合预算的规格；仍需满足用户其他要求。",
    )
    return result
