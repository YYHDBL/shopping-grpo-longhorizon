"""Aggregate calibration of the feed user's observable behavior model."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from shopping_grpo.feed.schema import EventType, UserEvent


_RATE_EVENTS = (
    EventType.WATCH,
    EventType.SKIP,
    EventType.LIKE,
    EventType.SHARE,
    EventType.CLICK,
    EventType.ADD_TO_CART,
    EventType.PURCHASE,
    EventType.RETURN,
)


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _counts(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, (list, tuple)):
        pairs = value
    else:
        raise TypeError("event_counts must be an object or sequence of pairs")
    normalized: dict[str, int] = {}
    for raw in pairs:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise TypeError("event_counts entries must be [event_type, count] pairs")
        event_type = UserEvent.from_dict({"event_type": raw[0]}).event_type
        count = raw[1]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"event count for {event_type.value!r} must be a non-negative integer")
        normalized[event_type.value] = count
    enum_order = {event.value: index for index, event in enumerate(EventType)}
    return tuple(sorted(normalized.items(), key=lambda item: enum_order[item[0]]))


@dataclass(frozen=True, slots=True)
class BehaviorCalibration:
    """Empirical transition probabilities for a numerical user simulator.

    ``cart_probability``, ``purchase_probability``, and ``return_probability``
    are conditional on click, cart, and purchase respectively.  The remaining
    probabilities are conditioned on an impression (likes/shares prefer watch as
    their denominator when watch events are present).
    """

    watch_probability: float
    skip_probability: float
    like_probability: float
    share_probability: float
    click_probability: float
    cart_probability: float
    purchase_probability: float
    return_probability: float
    mean_dwell_seconds: float
    dwell_std_seconds: float = 0.0
    mean_purchase_value: float = 0.0
    sample_size: int = 0
    event_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "watch_probability",
            "skip_probability",
            "like_probability",
            "share_probability",
            "click_probability",
            "cart_probability",
            "purchase_probability",
            "return_probability",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
        for name in ("mean_dwell_seconds", "dwell_std_seconds", "mean_purchase_value"):
            object.__setattr__(self, name, _non_negative(getattr(self, name), name=name))
        if isinstance(self.sample_size, bool) or not isinstance(self.sample_size, int):
            raise TypeError("sample_size must be an integer")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        object.__setattr__(self, "event_counts", _counts(self.event_counts))

    @classmethod
    def default(cls) -> "BehaviorCalibration":
        """Return explicit conservative priors used when an event denominator is absent."""

        return cls(
            watch_probability=0.55,
            skip_probability=0.45,
            like_probability=0.08,
            share_probability=0.02,
            click_probability=0.05,
            cart_probability=0.12,
            purchase_probability=0.28,
            return_probability=0.08,
            mean_dwell_seconds=8.0,
            dwell_std_seconds=4.0,
            mean_purchase_value=0.0,
        )

    @classmethod
    def from_events(
        cls,
        events: Iterable[UserEvent | Mapping[str, object]] | Mapping[str, object],
        *,
        prior: "BehaviorCalibration | None" = None,
        smoothing: float = 0.0,
    ) -> "BehaviorCalibration":
        """Calibrate from raw events or aggregate ``event_type -> count`` data.

        Aggregate mapping values may be integer counts, or objects containing
        ``count`` plus optional ``dwell_seconds``/``total_dwell_seconds`` and
        ``value``/``total_value``.  This makes KuaiRand/KuaiRec-style preprocessing
        possible without materializing millions of duplicate Python objects.
        """

        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
            raise TypeError("smoothing must be a number")
        smoothing = float(smoothing)
        if not math.isfinite(smoothing) or smoothing < 0.0:
            raise ValueError("smoothing must be finite and non-negative")
        prior = prior or cls.default()

        normalized_events: list[UserEvent] = []
        dwell_total_override = 0.0
        purchase_total_override = 0.0
        if isinstance(events, Mapping):
            if any(key in events for key in ("event_type", "type", "event")):
                normalized_events.append(UserEvent.from_dict(events))
            else:
                for raw_type, aggregate in events.items():
                    if isinstance(aggregate, bool):
                        raise TypeError(f"aggregate for {raw_type!r} must not be boolean")
                    if isinstance(aggregate, int):
                        if aggregate < 0:
                            raise ValueError(f"aggregate for {raw_type!r} must be non-negative")
                        if aggregate:
                            normalized_events.append(
                                UserEvent.from_dict({"event_type": raw_type, "count": aggregate})
                            )
                        continue
                    if not isinstance(aggregate, Mapping):
                        raise TypeError(
                            f"aggregate for {raw_type!r} must be an integer or object"
                        )
                    count = aggregate.get("count", 0)
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise ValueError(
                            f"aggregate count for {raw_type!r} must be a non-negative integer"
                        )
                    if not count:
                        continue
                    dwell = aggregate.get("dwell_seconds", aggregate.get("mean_dwell_seconds", 0.0))
                    value = aggregate.get("value", aggregate.get("mean_value", 0.0))
                    if "total_dwell_seconds" in aggregate:
                        dwell_total_override += _non_negative(
                            aggregate["total_dwell_seconds"],
                            name=f"{raw_type}.total_dwell_seconds",
                        )
                        dwell = 0.0
                    if "total_value" in aggregate:
                        purchase_total_override += _non_negative(
                            aggregate["total_value"], name=f"{raw_type}.total_value"
                        )
                        value = 0.0
                    normalized_events.append(
                        UserEvent.from_dict(
                            {
                                "event_type": raw_type,
                                "count": count,
                                "dwell_seconds": dwell,
                                "value": value,
                            }
                        )
                    )
        else:
            for event in events:
                normalized_events.append(
                    event if isinstance(event, UserEvent) else UserEvent.from_dict(event)
                )

        counts: dict[EventType, int] = defaultdict(int)
        dwell_values: list[tuple[float, int]] = []
        purchase_value = purchase_total_override
        for event in normalized_events:
            counts[event.event_type] += event.count
            if event.event_type is EventType.WATCH:
                dwell_values.append((event.dwell_seconds, event.count))
            if event.event_type is EventType.PURCHASE:
                purchase_value += event.value * event.count

        impressions = counts[EventType.IMPRESSION]
        if impressions == 0:
            impressions = max(
                counts[EventType.WATCH] + counts[EventType.SKIP],
                counts[EventType.CLICK],
                counts[EventType.LIKE],
                counts[EventType.SHARE],
            )
        watches = counts[EventType.WATCH]
        skips = counts[EventType.SKIP]
        watch_denominator = watches if watches else impressions

        def rate(numerator: int, denominator: int, fallback: float) -> float:
            if denominator <= 0:
                return fallback
            bounded = min(numerator, denominator)
            if smoothing:
                return (bounded + smoothing * fallback) / (denominator + smoothing)
            return bounded / denominator

        dwell_count = sum(count for _, count in dwell_values)
        dwell_total = dwell_total_override + math.fsum(
            value * count for value, count in dwell_values
        )
        mean_dwell = dwell_total / dwell_count if dwell_count else prior.mean_dwell_seconds
        if dwell_count:
            variance = math.fsum(
                count * (value - mean_dwell) ** 2 for value, count in dwell_values
            ) / dwell_count
            dwell_std = math.sqrt(max(variance, 0.0))
        else:
            dwell_std = prior.dwell_std_seconds

        purchases = counts[EventType.PURCHASE]
        return cls(
            watch_probability=rate(watches, impressions, prior.watch_probability),
            skip_probability=rate(skips, impressions, prior.skip_probability),
            like_probability=rate(
                counts[EventType.LIKE], watch_denominator, prior.like_probability
            ),
            share_probability=rate(
                counts[EventType.SHARE], watch_denominator, prior.share_probability
            ),
            click_probability=rate(
                counts[EventType.CLICK], impressions, prior.click_probability
            ),
            cart_probability=rate(
                counts[EventType.ADD_TO_CART],
                counts[EventType.CLICK],
                prior.cart_probability,
            ),
            purchase_probability=rate(
                purchases,
                counts[EventType.ADD_TO_CART],
                prior.purchase_probability,
            ),
            return_probability=rate(
                counts[EventType.RETURN], purchases, prior.return_probability
            ),
            mean_dwell_seconds=mean_dwell,
            dwell_std_seconds=dwell_std,
            mean_purchase_value=(
                purchase_value / purchases if purchases else prior.mean_purchase_value
            ),
            sample_size=impressions,
            event_counts=tuple(
                (event.value, counts[event]) for event in EventType if counts[event]
            ),
        )

    @classmethod
    def from_aggregates(
        cls,
        aggregates: Mapping[str, object],
        *,
        prior: "BehaviorCalibration | None" = None,
        smoothing: float = 0.0,
    ) -> "BehaviorCalibration":
        return cls.from_events(aggregates, prior=prior, smoothing=smoothing)

    def event_count(self, event_type: EventType | str) -> int:
        parsed = UserEvent.from_dict({"event_type": event_type}).event_type.value
        return dict(self.event_counts).get(parsed, 0)

    def probability(self, event_type: EventType | str) -> float:
        parsed = UserEvent.from_dict({"event_type": event_type}).event_type
        lookup = {
            EventType.WATCH: self.watch_probability,
            EventType.SKIP: self.skip_probability,
            EventType.LIKE: self.like_probability,
            EventType.SHARE: self.share_probability,
            EventType.CLICK: self.click_probability,
            EventType.ADD_TO_CART: self.cart_probability,
            EventType.PURCHASE: self.purchase_probability,
            EventType.RETURN: self.return_probability,
        }
        if parsed not in lookup:
            raise ValueError(f"{parsed.value!r} does not have a calibrated probability")
        return lookup[parsed]

    @property
    def watch_rate(self) -> float:
        return self.watch_probability

    @property
    def skip_rate(self) -> float:
        return self.skip_probability

    @property
    def like_rate(self) -> float:
        return self.like_probability

    @property
    def share_rate(self) -> float:
        return self.share_probability

    @property
    def click_rate(self) -> float:
        return self.click_probability

    @property
    def cart_rate(self) -> float:
        return self.cart_probability

    @property
    def purchase_rate(self) -> float:
        return self.purchase_probability

    @property
    def return_rate(self) -> float:
        return self.return_probability

    def to_dict(self) -> dict[str, object]:
        return {
            "watch_probability": self.watch_probability,
            "skip_probability": self.skip_probability,
            "like_probability": self.like_probability,
            "share_probability": self.share_probability,
            "click_probability": self.click_probability,
            "cart_probability": self.cart_probability,
            "purchase_probability": self.purchase_probability,
            "return_probability": self.return_probability,
            "mean_dwell_seconds": self.mean_dwell_seconds,
            "dwell_std_seconds": self.dwell_std_seconds,
            "mean_purchase_value": self.mean_purchase_value,
            "sample_size": self.sample_size,
            "event_counts": dict(self.event_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BehaviorCalibration":
        if not isinstance(data, Mapping):
            raise TypeError("behavior calibration must be an object")

        def value(probability_name: str, rate_name: str, fallback: float) -> object:
            return data.get(probability_name, data.get(rate_name, fallback))

        defaults = cls.default()
        sample_size = data.get("sample_size", 0)
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise TypeError("sample_size must be an integer")
        return cls(
            watch_probability=_probability(
                value("watch_probability", "watch_rate", defaults.watch_probability),
                name="watch_probability",
            ),
            skip_probability=_probability(
                value("skip_probability", "skip_rate", defaults.skip_probability),
                name="skip_probability",
            ),
            like_probability=_probability(
                value("like_probability", "like_rate", defaults.like_probability),
                name="like_probability",
            ),
            share_probability=_probability(
                value("share_probability", "share_rate", defaults.share_probability),
                name="share_probability",
            ),
            click_probability=_probability(
                value("click_probability", "click_rate", defaults.click_probability),
                name="click_probability",
            ),
            cart_probability=_probability(
                value("cart_probability", "cart_rate", defaults.cart_probability),
                name="cart_probability",
            ),
            purchase_probability=_probability(
                value("purchase_probability", "purchase_rate", defaults.purchase_probability),
                name="purchase_probability",
            ),
            return_probability=_probability(
                value("return_probability", "return_rate", defaults.return_probability),
                name="return_probability",
            ),
            mean_dwell_seconds=_non_negative(
                data.get("mean_dwell_seconds", defaults.mean_dwell_seconds),
                name="mean_dwell_seconds",
            ),
            dwell_std_seconds=_non_negative(
                data.get("dwell_std_seconds", defaults.dwell_std_seconds),
                name="dwell_std_seconds",
            ),
            mean_purchase_value=_non_negative(
                data.get("mean_purchase_value", defaults.mean_purchase_value),
                name="mean_purchase_value",
            ),
            sample_size=sample_size,
            event_counts=_counts(data.get("event_counts", {})),
        )


def calibrate_behavior(
    events: Iterable[UserEvent | Mapping[str, object]] | Mapping[str, object],
    *,
    prior: BehaviorCalibration | None = None,
    smoothing: float = 0.0,
) -> BehaviorCalibration:
    """Functional alias for callers that do not need the class constructor."""

    return BehaviorCalibration.from_events(events, prior=prior, smoothing=smoothing)


__all__ = ["BehaviorCalibration", "calibrate_behavior"]
