"""Action-invariant random numbers for reproducible Feed counterfactuals.

The simulator never consumes a mutable random stream.  Every exogenous draw is
addressed by episode, video step, channel and entity, so two rollouts that only
change one intervention receive exactly the same external noise everywhere else.
"""

from __future__ import annotations

import hashlib
import math


class CommonRandomNumbers:
    """Deterministic, stateless random-number table backed by SHA-256."""

    def __init__(self, seed: int | str, episode_id: str = "") -> None:
        self.seed = str(seed)
        self.episode_id = str(episode_id)

    def uniform(self, step: int, channel: str, entity: str = "") -> float:
        """Return a stable value in the open interval ``(0, 1)``."""
        address = "\x1f".join(
            (self.seed, self.episode_id, str(int(step)), str(channel), str(entity))
        )
        digest = hashlib.sha256(address.encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], "big", signed=False)
        return (integer + 0.5) / (2**64)

    def normal(self, step: int, channel: str, entity: str = "") -> float:
        """Return a stable standard-normal draw using Box-Muller."""
        first = self.uniform(step, f"{channel}:u1", entity)
        second = self.uniform(step, f"{channel}:u2", entity)
        return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)

    def bernoulli(
        self,
        probability: float,
        step: int,
        channel: str,
        entity: str = "",
    ) -> bool:
        """Sample a Bernoulli outcome after clamping numerical drift."""
        probability = min(max(float(probability), 0.0), 1.0)
        return self.uniform(step, channel, entity) < probability

    def integer(
        self,
        low: int,
        high: int,
        step: int,
        channel: str,
        entity: str = "",
    ) -> int:
        """Return a stable integer from the inclusive range ``[low, high]``."""
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        width = high - low + 1
        return low + min(int(self.uniform(step, channel, entity) * width), width - 1)


def sigmoid(value: float) -> float:
    """Numerically stable logistic function used by the hybrid user model."""
    value = float(value)
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
