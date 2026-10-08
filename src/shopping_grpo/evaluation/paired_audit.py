"""Offline paired audit of saved collector trajectories; never runs a model."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def terminal_for_policy(terminal, policy="environment"):
    if policy not in {"environment", "original"}:
        raise ValueError("unknown reward policy")
    result = deepcopy(terminal)
    if policy == "environment" or not result.get("purchase"):
        return result
    detail = result.get("reward_detail") or {}
    legacy = (detail.get("evidence") or {}).get("legacy_result")
    if legacy is None:
        if (detail.get("evidence") or {}).get("review_policy_version"):
            raise ValueError("reviewed purchase is missing original reward")
        return result
    if not result.get("done") or not result.get("over"):
        raise ValueError("incomplete reviewed purchase")
    result["reviewed_reward_detail"] = detail
    result["reward"] = legacy["reward"]
    result["reward_detail"] = {
        **legacy,
        "reward_version": "shopsimulator-reward-v3",
        "terminal_utility": legacy["reward"],
        "termination_reason": legacy["reward_type"],
        "purchase_success": legacy["reward_valid"] is True
        and legacy["reward_type"] in {"gold_purchase", "valid_alternative_purchase"},
    }
    return result


def strict_success(trajectory, policy="original"):
    terminal = terminal_for_policy(trajectory.get("terminal_result") or {}, policy)
    detail = terminal.get("reward_detail") or {}
    return bool(
        trajectory.get("done")
        and terminal.get("done")
        and terminal.get("over")
        and terminal.get("purchase")
        and detail.get("reward_version") == "shopsimulator-reward-v3"
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
        and not trajectory.get("error")
        and not trajectory.get("release_error")
    )


def compare(original, enhanced, policy="original"):
    def indexed(rows):
        result = {}
        for row in rows:
            task = int(row["task_id"])
            if task in result:
                raise ValueError("duplicate task ID; audit one attempt per task")
            result[task] = strict_success(row, policy)
        return result

    left, right = indexed(original), indexed(enhanced)
    if not left or left.keys() != right.keys():
        raise ValueError("paired task sets must match and be nonempty")
    return {
        "total": len(left),
        "reward_policy": policy,
        "original_success": sum(left.values()),
        "enhanced_success": sum(right.values()),
        "gained": sorted(k for k in left if right[k] and not left[k]),
        "lost": sorted(k for k in left if left[k] and not right[k]),
        "interpretation": "Paired observations; not component-level causal attribution.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--enhanced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-policy", choices=["original", "environment"], default="original")
    args = parser.parse_args(argv)
    paths = [args.original.resolve(), args.enhanced.resolve(), args.output.resolve()]
    if len(set(paths)) != 3 or args.output.exists():
        parser.error("use distinct inputs and a new output")
    payloads = [p.read_bytes() for p in (args.original, args.enhanced)]
    rows = [[json.loads(line) for line in raw.splitlines() if line.strip()] for raw in payloads]
    report = compare(*rows, policy=args.reward_policy)
    report["input_sha256"] = [hashlib.sha256(raw).hexdigest() for raw in payloads]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
