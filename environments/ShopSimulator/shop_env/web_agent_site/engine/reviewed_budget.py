"""Public budget and payment-stage evidence; no invented approximation tolerance."""

import math
import re

from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE, comparison

VERSION = "public-budget-payment-v2"
N = r"(\d+(?:\.\d+)?)"
C = r"(?:元(?!素)|块钱|块(?![0-9一二三四五六七八九十]|磁盘|硬盘|存储|盘))"


def _normalize_money(text):
    def convert(m):
        raw = m[0]
        if re.fullmatch(r"[零〇一二三四五六七八九]+", raw):
            return str(
                int("".join(str("零一二三四五六七八九".index(c.replace("〇", "零"))) for c in raw))
            )
        # Reject colloquial dropped units (三百五) and noncanonical/repeated units.
        if re.search(r"[百千万][一二三四五六七八九]$", raw):
            return raw
        total = section = number = 0
        last = 10000
        for c in raw:
            if c in "零〇一二三四五六七八九两":
                number = 2 if c == "两" else "零一二三四五六七八九".index(c.replace("〇", "零"))
            else:
                unit = {"十": 10, "百": 100, "千": 1000, "万": 10000}[c]
                if unit == 10000:
                    total += (section + number) * unit
                    section = number = 0
                    last = 10000
                else:
                    if unit >= last:
                        return raw
                    section += (number or 1) * unit
                    number = 0
                    last = unit
        return str(total + section + number)

    return re.sub(
        r"(?<![\d.零〇一二三四五六七八九十百千万两])[零〇一二三四五六七八九十百千万两]+(?=\s*(?:元(?!素)|块钱|块(?:左右|上下|以内|以下|以上|$)))",
        convert,
        text,
    )


def _base_budget(instruction):
    lower, upper, approximate, evidence = [], [], [], []
    price_language = False
    for clause in re.split(r"[，,。；;！!？?\n]", str(instruction)):
        original_clause = clause
        clause = _normalize_money(clause)
        monetary = bool(
            re.search(r"预算|价格|价位|总价|售价|花费", clause) or re.search(rf"{N}\s*{C}", clause)
        )
        price_language |= monetary
        if not monetary:
            continue
        if re.search(r"左右|上下|大概|大约|差不多|约莫|(?:^|预算|价格|在)约", clause):
            approximate.append(original_clause)
            # Preserve explicit hard ceilings even beside approximate wording.
            for m in re.finditer(
                rf"(?:不超过|不要超过|不得超过|不能超过|不高于)\s*{N}\s*{C}", clause
            ):
                upper.append(float(m[1]))
                evidence.append(m[0])
            continue
        patterns = [
            (
                "range",
                rf"(?:预算|价格|价位|总价)(?:要|需要|控制)?(?:在|为|是)?\s*{N}\s*元?\s*[-—–~～至到]\s*{N}\s*{C}",
            ),
            ("lower", rf"(?:不少于|不低于|至少)\s*{N}\s*{C}"),
            ("lower", rf"{N}\s*{C}\s*以上"),
            (
                "upper",
                rf"(?:不超过|不要超过|不得超过|不能超过|别超过|不高于|最多|至多|不超)\s*{N}\s*{C}",
            ),
            ("upper", rf"{N}\s*{C}\s*(?:以内|以下)"),
        ]
        matched = False
        for kind, pattern in patterns:
            for m in re.finditer(pattern, clause):
                matched = True
                evidence.append(m[0])
                if kind == "range":
                    lower.append(float(m[1]))
                    upper.append(float(m[2]))
                elif kind == "lower":
                    lower.append(float(m[1]))
                else:
                    upper.append(float(m[1]))
        if not matched:
            m = re.search(
                rf"(?:预算|总价|价格控制)(?:控制在|在|为|是|有)?\s*{N}\s*{C}(?:$|[之内])", clause
            )
            if m:
                upper.append(float(m[1]))
                evidence.append(m[0])
                matched = True
        if not matched:
            # Unsupported wording, including Chinese numerals, is not silently passed.
            approximate.append(original_clause)
    return {
        "lower": max(lower) if lower else None,
        "upper": min(upper) if upper else None,
        "unresolved": approximate,
        "price_language": price_language,
        "evidence": evidence,
    }


def _money_amounts(clause):
    normalized = _normalize_money(clause)
    normalized = re.sub(
        r"(\d+(?:\.\d+)?)(万|千)(?=元|块)",
        lambda m: str(float(m[1]) * (10000 if m[2] == "万" else 1000)),
        normalized,
    )
    amounts = [
        float(m[1])
        for m in re.finditer(rf"(?<![\d.一二三四五六七八九十百千万两]){N}\s*{C}", normalized)
    ]
    if not amounts:
        m = re.search(
            r"(?:预算|价格|价位|售价|总价)(?:控制|可以接受|大致|大约|大概|差不多|接近|最好|在|为|是|约)*\s*(\d+(?:\.\d+)?)(?:左右|上下)",
            normalized,
        )
        if m:
            amounts = [float(m[1])]
    return amounts


def public_budget(instruction):
    result = _base_budget(instruction)
    result["approximate_targets"] = []
    result["unsupported"] = []
    for clause in result["unresolved"]:
        numbers = _money_amounts(clause)
        if len(numbers) == 1 and re.search(
            r"左右|上下|大概|大约|差不多|接近|约莫|(?:^|预算|价格|在)约", clause
        ):
            result["approximate_targets"].append(
                {"amount": numbers[0], "text": clause, "tolerance": None}
            )
        elif re.fullmatch(r"准备\s*\d+(?:\.\d+)?元应该够了", clause.strip()):
            # Explicit available funds, not a price target. Keep the interpretation visible.
            amount = numbers[0]
            result["upper"] = (
                min(result["upper"], amount) if result["upper"] is not None else amount
            )
            result["evidence"].append(
                {
                    "text": clause,
                    "interpretation": "stated_available_funds_ceiling",
                    "amount": amount,
                }
            )
        else:
            result["unsupported"].append(clause)
    return result


def budget_gate(price_resolution, goal, selected_options=None):
    text = goal.get("instruction_text", "")
    bounds = public_budget(text)
    price = price_resolution.get("price") if isinstance(price_resolution, dict) else None
    try:
        valid = (
            price_resolution.get("status") == PASS
            and math.isfinite(float(price))
            and float(price) >= 0
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    lo, hi = bounds["lower"], bounds["upper"]
    payment = None
    if "定金" in text:
        selected = " | ".join(str(v) for v in (selected_options or {}).values())
        requested = re.findall(r"定金(?:可以接受|接受|不超过|为|是)?\s*(\d+(?:\.\d+)?)元", text)
        quoted = re.findall(r"定金\s*(\d+(?:\.\d+)?)元", selected)
        payment = {
            "stage": "deposit",
            "requested_amounts": requested,
            "selected_text": selected,
            "quoted_deposits": quoted,
            "charged_price": price,
            "full_price_not_assumed": True,
        }
        if valid and "定金" in selected and len(quoted) == 1 and float(quoted[0]) != float(price):
            payment["status"] = "data_conflict"
        elif (
            valid
            and "定金" in selected
            and len(requested) == 1
            and "全款" not in text
            and float(requested[0]) == float(price)
        ):
            payment["status"] = "verified_deposit_amount"
        else:
            payment["status"] = "unresolved_payment_stage_or_amount"
        status = PASS if payment["status"] == "verified_deposit_amount" else UNVERIFIABLE
    elif not valid or (lo is not None and hi is not None and lo > hi):
        status = UNVERIFIABLE
    elif (lo is not None and float(price) < lo) or (hi is not None and float(price) > hi):
        status = FAIL
    elif bounds["unsupported"]:
        status = UNVERIFIABLE
    elif bounds["approximate_targets"]:
        # Zero deviation satisfies every nonnegative tolerance; never pick a tolerance.
        status = (
            PASS
            if all(
                math.isclose(float(price), t["amount"], rel_tol=0, abs_tol=1e-9)
                for t in bounds["approximate_targets"]
            )
            else UNVERIFIABLE
        )
    else:
        status = PASS
    distances = []
    if valid and payment is None:
        for target in bounds["approximate_targets"]:
            amount = target["amount"]
            delta = float(price) - amount
            distances.append(
                {
                    "target": amount,
                    "price": float(price),
                    "delta": delta,
                    "relative_deviation": abs(delta) / amount if amount else None,
                    "exact_center": abs(delta) < 1e-9,
                    "sensitivity_only": {
                        str(t): abs(delta) <= amount * t + 1e-9 for t in [0, 0.01, 0.05, 0.10, 0.20]
                    },
                    "note": "Hypothetical symmetric bands are diagnostics, not user-authorized tolerances.",
                }
            )
    return comparison(
        status,
        comparator=VERSION,
        required=bounds,
        actual=price,
        source_field="instruction_text|variant_price|selected_options",
        evidence={
            "price_resolution": price_resolution,
            "payment": payment,
            "approximation_diagnostics": distances,
        },
    )
