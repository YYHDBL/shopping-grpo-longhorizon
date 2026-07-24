"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (uid, raw_reward) in enumerate(zip(uids, seq_rewards, strict=True)):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")

        group = grouped.setdefault(uid, {"uid": uid, "indices": [], "rewards": []})
        group["indices"].append(index)
        group["rewards"].append(reward)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        rewards = group["rewards"]
        reward_min = min(rewards)
        reward_max = max(rewards)
        keep = reward_max - reward_min > tolerance
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(rewards),
                "reward_min": reward_min,
                "reward_max": reward_max,
                "kept": keep,
            }
        )

    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats
