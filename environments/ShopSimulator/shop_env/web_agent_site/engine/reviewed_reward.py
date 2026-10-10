"""Opt-in Reward v3 review policy. Public budgets and uncertainty stay explicit.

No target requirements are guessed from missing text. Original scorer stays available.
"""

import re
from copy import deepcopy

from web_agent_site.engine import reward as legacy
from web_agent_site.engine.comparators import FAIL, PASS, UNVERIFIABLE

VERSION = "shopsimulator-reward-v3-review6"
from web_agent_site.engine.reviewed_budget import budget_gate
from web_agent_site.engine.reviewed_public_constraints import product_checks, selected_checks


def _preferences(product, goal, selected_options):
    preferences = deepcopy(legacy._preference_dimensions(product, goal, selected_options))
    for r in preferences["dimensions"]["core_functions"]["results"]:
        # Narrow reviewed linguistic equivalence; preserve polarity checks.
        aliases = {"推车架子": "推车架", "防结露": "防凝露", "官方正品": "官网正品"}
        required = r.get("required") or []
        if (
            len(required) == 1
            and required[0] in aliases
            and r["status"] != PASS
            and not (r.get("evidence") or {}).get("negated")
        ):
            original_required = deepcopy(required)
            alias_result = legacy.compare_core_functions([aliases[required[0]]], product)
            r.update(alias_result)
            r["required"] = original_required
            r.setdefault("evidence", {})["reviewed_alias"] = {required[0]: aliases[required[0]]}
        evidence = r.get("evidence") or {}
        if evidence.get("missing") and not evidence.get("negated"):
            r.update(status=UNVERIFIABLE, passed=False, verifiable=False)
            r["comparator"] += "-missing-is-unknown"
    # Only ignore a reviewed installation qualifier when the instruction does not
    # request it and every other character of the selected value is identical.
    if not re.search(r"孔|钩|安装", str(goal.get("instruction_text", ""))):
        for r in preferences["dimensions"]["key_options"]["results"]:
            e = r.get("evidence") or {}
            wrong = e.get("wrong_values") or []
            strip = lambda value: re.sub(r"打孔|挂钩", "", str(value))
            if (
                wrong
                and not e.get("missing_axes")
                and not e.get("unavailable_values")
                and all(
                    {
                        (
                            "打孔"
                            if "打孔" in str(w.get(k, ""))
                            else "挂钩"
                            if "挂钩" in str(w.get(k, ""))
                            else ""
                        )
                        for k in ["required", "selected"]
                    }
                    == {"打孔", "挂钩"}
                    and strip(w["required"]) == strip(w["selected"])
                    for w in wrong
                )
            ):
                r.update(status=PASS, passed=True, verifiable=True)
                e["ignored_unrequested_installation"] = deepcopy(wrong)
                r["comparator"] += "-unrequested-installation-reviewed"
    # An unrequested style-code suffix is not a customer requirement. Remove only
    # the suffix when every other character of the two public option values agrees.
    instruction = str(goal.get("instruction_text", "")).casefold()
    for r in preferences["dimensions"]["key_options"]["results"]:
        e = r.get("evidence") or {}
        wrong = e.get("wrong_values") or []
        if not wrong or e.get("missing_axes") or e.get("unavailable_values"):
            continue
        equivalent = True
        for w in wrong:
            left = re.fullmatch(r"(.*?)\s+(\d+[a-z]+)款", str(w.get("required", "")).casefold())
            right = re.fullmatch(r"(.*?)\s+(\d+[a-z]+)款", str(w.get("selected", "")).casefold())
            if (
                not left
                or not right
                or left[1] != right[1]
                or left[2] in instruction
                or right[2] in instruction
            ):
                equivalent = False
                break
        if equivalent:
            r.update(status=PASS, passed=True, verifiable=True)
            e["ignored_unrequested_style_codes"] = deepcopy(wrong)
            r["comparator"] += "-unrequested-style-code-reviewed"
    # Reviewed narrow lexical aliases; no substring deletion across other models.
    for r in preferences["dimensions"]["model"]["results"]:
        required = r.get("required") or []
        if required and all(str(v).casefold() in ("m2", "m.2") for v in required):
            normalized = dict(product)
            for field in ["title", "model"]:
                if field in normalized:
                    normalized[field] = re.sub(
                        r"(?i)(?<![a-z0-9])m\.2(?![a-z0-9])", "m2", str(normalized[field])
                    )
            old_required = deepcopy(required)
            r.update(legacy.compare_model(["m2"], normalized))
            r["required"] = old_required
            r["comparator"] += "-m2-interface-alias"
    # Require a selected-image shipment statement, not a photo of accessories.
    from web_agent_site.engine.public_option_evidence import selected_option_evidence

    images = selected_option_evidence(product, selected_options)
    for r in preferences["dimensions"]["core_functions"]["results"]:
        if r.get("required") != ["附带配件"] or (r.get("evidence") or {}).get("negated"):
            continue
        proofs = [
            i
            for i in images
            if i["facts"].get("shipment_scope_explicit") is True
            and "配件" in i["facts"].get("included_components", [])
        ]
        if proofs:
            r.update(status=PASS, passed=True, verifiable=True)
            r["comparator"] += "-selected-image-shipment"
            r["evidence"] = {
                "matched": [{"value": "附带配件", "source": "selected_option_image"}],
                "missing": [],
                "negated": [],
                "selected_image_evidence": proofs,
            }
    # Recompute coverage. A missing function is not proof it is absent.
    active_weight = preferences["active_weight"]
    for d in preferences["dimensions"].values():
        total = d["required_count"]
        d["passed_count"] = sum(r["status"] == PASS for r in d["results"])
        d["verifiable_count"] = sum(r["status"] != UNVERIFIABLE for r in d["results"])
        d["score"] = d["passed_count"] / total if total else 0
        d["coverage"] = d["verifiable_count"] / total if total else 0
    for field, dimension_field in [("match_score", "score"), ("evidence_coverage", "coverage")]:
        preferences[field] = (
            sum(
                legacy.DIMENSION_WEIGHTS[n] * d[dimension_field]
                for n, d in preferences["dimensions"].items()
                if d["active"]
            )
            / active_weight
            if active_weight
            else 1.0
        )
    preferences["all_satisfied"] = all(
        r["status"] == PASS for d in preferences["dimensions"].values() for r in d["results"]
    )
    return preferences


def evaluate_purchase(
    product, goal, *, selected_options, price_resolution=None, price=None, rewards=None
):
    old = legacy.evaluate_purchase(
        product,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
        price=price,
        rewards=rewards,
    )
    values = {**legacy.DEFAULT_REWARDS, **(rewards or {})}
    resolution = old.evidence["price_resolution"]
    gates = deepcopy(old.hard_gates)
    gates["budget"] = budget_gate(resolution, goal, selected_options)
    preferences = _preferences(product, goal, selected_options)
    public_checks = selected_checks(
        goal.get("instruction_text", ""), selected_options
    ) + product_checks(product, goal.get("instruction_text", ""), selected_options)
    checks = (
        public_checks
        + list(gates.values())
        + [r for d in preferences["dimensions"].values() for r in d["results"]]
    )
    # A proven hard conflict remains a failure even if another field is unknown.
    if any(r["status"] == FAIL for r in list(gates.values()) + public_checks):
        kind = "wrong_purchase"
        valid = True
    elif any(r["status"] == UNVERIFIABLE for r in checks):
        kind = "reward_unverifiable"
        valid = False
    elif preferences["all_satisfied"]:
        kind = "gold_purchase" if old.target_asin_match else "valid_alternative_purchase"
        valid = True
    else:
        kind = "partial_alternative_purchase"
        valid = True
    value = values.get(kind)
    if kind == "partial_alternative_purchase":
        value = min(
            values["partial_purchase_cap"],
            values["partial_purchase_base"]
            + values["partial_purchase_scale"] * preferences["match_score"],
        )
    evidence = {
        **old.evidence,
        "preference_scoring": preferences,
        "review_policy_version": VERSION,
        "public_constraint_checks": public_checks,
        "legacy_result": old.to_dict(),
        "limitations": [
            "Task-authored function/option annotations are not automatically verified against natural-language requirements.",
            "No full semantic feasibility or selected-variant function proof is inferred.",
        ],
    }
    return legacy.RewardResult(
        float(value),
        kind,
        valid,
        kind,
        old.target_asin_match,
        gates,
        preferences["match_score"],
        evidence,
    )


def evaluate_candidate_eligibility(product, goal):
    selected, option_resolution = legacy.candidate_options_for_evaluation(
        product, goal.get("required_options_by_key")
    )
    result = evaluate_purchase(product, goal, selected_options=selected)
    p = result.evidence["preference_scoring"]
    known = (
        all(c["status"] == PASS for c in result.evidence.get("public_constraint_checks", []))
        and all(g["status"] == PASS for g in result.hard_gates.values())
        and p["all_satisfied"]
    )
    return {
        "status": PASS if known else FAIL,
        "known_acceptable": known,
        "known_valid": known,
        "selected_options": selected,
        "option_resolution": option_resolution,
        "price_resolution": result.evidence["price_resolution"],
        "hard_gates": result.hard_gates,
        "match_score": p["match_score"],
        "evidence_coverage": p["evidence_coverage"],
        "review_policy_version": VERSION,
    }
