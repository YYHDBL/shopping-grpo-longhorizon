"""Select the frozen environment's original Reward v3 for learning only."""

import math
from copy import deepcopy


def select_legacy_reward(output, events):
    info = output.extra_fields["shopping"]
    if not info["done"] or info.get("infrastructure_invalid"):
        return
    purchases = [e["result"] for e in events if e.get("result", {}).get("purchase")]
    if not purchases:
        return
    terminal = purchases[-1]
    if not terminal.get("done") or not terminal.get("over"):
        raise RuntimeError("Incomplete purchase terminal result")
    detail = terminal.get("reward_detail") or {}
    legacy = (detail.get("evidence") or {}).get("legacy_result")
    if legacy is None:
        raise RuntimeError("Frozen purchase is missing its original Reward v3 result")
    value = float(legacy["reward"])
    if not math.isfinite(value) or not isinstance(legacy["reward_valid"], bool):
        raise RuntimeError("Invalid original Reward v3 result")
    valid = legacy["reward_valid"]
    kind = legacy["reward_type"]
    info["review6_diagnostics"] = {
        "reward": deepcopy(info["reward"]),
        "reward_type": info["reward_type"],
        "reward_valid": info["reward_valid"],
    }
    info.update(
        reward_type=kind,
        reward_valid=valid,
        reward_unverifiable=not valid,
        selected_reward_policy="original-reward-v3",
    )
    # Auxiliary dimensions remain audit-only review6 values; all learning and
    # success/filtering fields below come from the original score.
    info["reward"].update(
        total=value if valid else 0.0,
        terminal_utility=value if valid else 0.0,
        native=value,
        full=float(kind == "gold_purchase" and valid),
        strict=float(kind == "gold_purchase" and valid),
        semantic=float(kind in {"gold_purchase", "valid_alternative_purchase"} and valid),
        purchase_success=float(kind in {"gold_purchase", "valid_alternative_purchase"} and valid),
        sampling_invalid=not valid,
        reward_unverifiable=not valid,
    )
    info["auxiliary_dimensions_policy"] = "review6_diagnostic_only"
    output.reward_score = value if valid else 0.0
