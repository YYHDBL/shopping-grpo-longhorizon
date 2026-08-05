"""Dependency-free offline metrics for long-horizon Feed episodes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from shopping_grpo.feed.schema import EpisodeResult


def _row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    if is_dataclass(value):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return []


def _transitions(episode: Any) -> list[dict[str, Any]]:
    if isinstance(episode, EpisodeResult):
        source: Any = episode.transitions
    elif isinstance(episode, Mapping):
        source = episode.get("transitions", episode.get("trajectory", ()))
    elif hasattr(episode, "transitions"):
        source = getattr(episode, "transitions")
    elif isinstance(episode, Sequence) and not isinstance(episode, (str, bytes)):
        source = episode
    else:
        raise TypeError("episode must be an EpisodeResult, mapping, or transition sequence")
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise TypeError("episode transitions must be a sequence")
    return [_row(item) for item in source]


def _raw_events(transition: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = _row(transition.get("metadata"))
    events = metadata.get("raw_events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        events = transition.get("events", transition.get("user_events", ()))
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return []
    return [_row(event) for event in events]


def _event_type(event: Mapping[str, Any]) -> str:
    value = event.get("event_type", event.get("type", event.get("event", "")))
    value = getattr(value, "value", value)
    aliases = {"cart": "add_to_cart", "buy": "purchase", "refund": "return"}
    return aliases.get(str(value), str(value))


def _event_count(event: Mapping[str, Any]) -> int:
    count = event.get("count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return 1
    return count


def _event_source(event: Mapping[str, Any], default: int) -> int:
    metadata = _row(event.get("metadata"))
    source = event.get("source_step", metadata.get("source_step", default))
    if isinstance(source, bool) or not isinstance(source, int) or source < 0:
        return default
    return source


def _reward(transition: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _row(transition.get("metadata"))
    raw = metadata.get("raw_reward_breakdown")
    return dict(raw) if isinstance(raw, Mapping) else _row(transition.get("reward"))


def _reward_total(transition: Mapping[str, Any]) -> float:
    reward = _reward(transition)
    if "total" in reward:
        return _number(reward["total"])
    return math.fsum(_number(value) for value in reward.values())


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _summary(episode: Any) -> dict[str, Any]:
    root = _row(episode)
    final_state = _row(root.get("final_state"))
    for candidate in (
        root.get("evaluator_summary"),
        final_state.get("evaluation_summary"),
        root.get("summary"),
        root.get("metrics"),
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def evaluate_episode(episode: Any) -> dict[str, Any]:
    """Compute conversion, experience, grounding, and long-term value metrics."""

    transitions = _transitions(episode)
    root = _row(episode)
    summary = _summary(episode)
    events_by_transition = [_raw_events(transition) for transition in transitions]

    actions = [_row(transition.get("action")) for transition in transitions]
    interventions = sum(
        1 for action in actions if str(action.get("decision")) == "recommend"
    )
    no_recommendations = sum(
        1 for action in actions if str(action.get("decision")) == "no_recommend"
    )
    grounded_indices: set[int] = set()
    for index, action in enumerate(actions):
        if str(action.get("decision")) != "recommend":
            continue
        evidence = _strings(action.get("evidence_ids"))
        if any(item.startswith("product.") for item in evidence) and any(
            item.startswith(("video.", "history.", "persona.")) for item in evidence
        ):
            grounded_indices.add(index)
    grounded = len(grounded_indices)

    event_counts: dict[str, int] = {}
    purchase_events: list[tuple[int, dict[str, Any]]] = []
    return_events: list[tuple[int, dict[str, Any]]] = []
    correct_no_recommend = 0
    dwell_seconds = 0.0
    complementary_bundle_offers = 0
    for index, events in enumerate(events_by_transition):
        step = transitions[index].get("step", index)
        if isinstance(step, bool) or not isinstance(step, int):
            step = index
        for event in events:
            name = _event_type(event)
            count = _event_count(event)
            event_counts[name] = event_counts.get(name, 0) + count
            if name in {"watch", "skip"}:
                dwell_seconds += _number(
                    event.get("dwell_seconds", event.get("value"))
                ) * count
            if name == "purchase":
                purchase_events.append((step, event))
            elif name == "return":
                return_events.append((step, event))
            elif name == "bundle_offer" and bool(
                _row(event.get("metadata")).get("complementary")
            ):
                complementary_bundle_offers += count
            elif name == "no_recommend" and bool(
                _row(event.get("metadata")).get("hindsight_correct")
            ):
                correct_no_recommend += count

    # Environments that omit explicit no-recommend events can still expose the
    # signed diagnostic component in the raw reward breakdown.
    if correct_no_recommend == 0:
        correct_no_recommend = sum(
            1
            for transition, action in zip(transitions, actions)
            if str(action.get("decision")) == "no_recommend"
            and _number(_reward(transition).get("correct_no_recommend")) > 0.0
        )

    purchases_summary = summary.get("purchases")
    if isinstance(purchases_summary, Sequence) and not isinstance(
        purchases_summary, (str, bytes)
    ):
        purchase_count = len(purchases_summary)
        return_count = sum(1 for record in purchases_summary if bool(_row(record).get("returned")))
        qualified_count = sum(
            1
            for record in purchases_summary
            if bool(_row(record).get("hard_match"))
            and _number(_row(record).get("soft_satisfaction")) >= 0.5
            and not bool(_row(record).get("returned"))
        )
    else:
        purchase_count = sum(_event_count(event) for _, event in purchase_events)
        return_count = sum(_event_count(event) for _, event in return_events)
        returned_keys = {
            (
                str(event.get("product_id", "")),
                _event_source(event, realized_step),
            )
            for realized_step, event in return_events
        }
        qualified_count = sum(
            _event_count(event)
            for realized_step, event in purchase_events
            if bool(_row(event.get("metadata")).get("qualified"))
            and (
                str(event.get("product_id", "")),
                _event_source(event, realized_step),
            )
            not in returned_keys
        )

    irrelevant = sum(
        1
        for transition, action in zip(transitions, actions)
        if str(action.get("decision")) == "recommend"
        and _number(
            _reward(transition).get(
                "irrelevant_recommendation",
                _reward(transition).get("irrelevance_penalty"),
            )
        )
        < 0.0
    )
    repeat = event_counts.get("repeat_exposure", 0)
    if repeat == 0:
        repeat = sum(
            1
            for transition in transitions
            if _number(_reward(transition).get("repeat_exposure")) < 0.0
        )
    interruptions = sum(
        1
        for transition in transitions
        if _number(
            _reward(transition).get(
                "interruption", _reward(transition).get("interruption_penalty")
            )
        )
        < 0.0
    )

    unsupported_indices = {
        index
        for index, (transition, action) in enumerate(zip(transitions, actions))
        if str(action.get("decision")) == "recommend"
        and (
            index not in grounded_indices
            or _number(
                _reward(transition).get(
                    "unsupported_claim",
                    _reward(transition).get("unsupported_claim_penalty"),
                )
            )
            < 0.0
        )
    }
    tool_calls = 0
    for transition in transitions:
        records = transition.get("tool_records", transition.get("tools"))
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            records = _row(_row(transition.get("metadata")).get("runtime_info")).get(
                "tool_records", ()
            )
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            tool_calls += len(records)

    bundle_action_indices = {
        index
        for index, action in enumerate(actions)
        if str(action.get("decision")) == "recommend"
        and (
            str(action.get("surface")) == "bundle"
            or str(action.get("strategy")) == "bundle"
            or str(action.get("relationship")) == "bundle"
            or len(_strings(action.get("product_ids"))) > 1
        )
    }
    bundle_event_indices: set[int] = set()
    complementary_bundle_indices: set[int] = set()
    for index, events in enumerate(events_by_transition):
        for event in events:
            if _event_type(event) != "bundle_offer":
                continue
            bundle_event_indices.add(index)
            if bool(_row(event.get("metadata")).get("complementary")):
                complementary_bundle_indices.add(index)
    for index, transition in enumerate(transitions):
        if _number(_reward(transition).get("bundle_value")) > 0.0:
            complementary_bundle_indices.add(index)
    bundle_offer_count = max(
        event_counts.get("bundle_offer", 0),
        len(bundle_action_indices | bundle_event_indices),
    )
    complementary_bundle_count = max(
        complementary_bundle_offers, len(complementary_bundle_indices)
    )

    click_count = event_counts.get("click", 0)
    cart_count = event_counts.get("add_to_cart", 0)
    watch_count = event_counts.get("watch", 0)
    skip_count = event_counts.get("skip", 0)
    like_count = event_counts.get("like", 0)
    steps = len(transitions)
    gross_revenue = math.fsum(
        _number(event.get("value")) * _event_count(event)
        for _, event in purchase_events
    )
    returned_revenue = math.fsum(
        _number(event.get("value")) * _event_count(event)
        for _, event in return_events
    )
    net_revenue = _number(summary.get("net_revenue"), gross_revenue - returned_revenue)
    fatigue = _number(
        summary.get(
            "terminal_fatigue",
            _row(root.get("final_state")).get("terminal_fatigue", 0.0),
        )
    )
    terminal_satisfaction = _number(
        summary.get(
            "terminal_satisfaction",
            _row(root.get("final_state")).get("terminal_satisfaction", 0.0),
        )
    )
    total_reward = (
        _number(root.get("total_reward"))
        if "total_reward" in root and root.get("total_reward") is not None
        else math.fsum(_reward_total(transition) for transition in transitions)
    )

    metrics: dict[str, Any] = {
        "episode_id": str(root.get("episode_id", summary.get("episode_id", ""))),
        "steps": steps,
        "interventions": interventions,
        "no_recommendations": no_recommendations,
        "clicks": click_count,
        "add_to_carts": cart_count,
        "watches": watch_count,
        "skips": skip_count,
        "likes": like_count,
        "total_dwell_seconds": dwell_seconds,
        "purchases": purchase_count,
        "qualified_purchases": qualified_count,
        "returns": return_count,
        "correct_no_recommendations": correct_no_recommend,
        "irrelevant_recommendations": irrelevant,
        "repeat_exposures": repeat,
        "interruptions": interruptions,
        "grounded_recommendations": grounded,
        "unsupported_claims": len(unsupported_indices),
        "bundle_offers": bundle_offer_count,
        "complementary_bundle_offers": complementary_bundle_count,
        "tool_calls": tool_calls,
        "click_rate": _safe_rate(click_count, interventions),
        "add_to_cart_rate": _safe_rate(cart_count, interventions),
        "purchase_rate": _safe_rate(purchase_count, interventions),
        "click_to_cart_rate": _safe_rate(cart_count, click_count),
        "cart_to_purchase_rate": _safe_rate(purchase_count, cart_count),
        "qualified_purchase_rate": _safe_rate(qualified_count, purchase_count),
        "correct_no_recommend_rate": _safe_rate(correct_no_recommend, no_recommendations),
        "interventions_per_100": _safe_rate(100.0 * interventions, steps),
        "return_rate": _safe_rate(return_count, purchase_count),
        "irrelevant_recommendation_rate": _safe_rate(irrelevant, interventions),
        "repeat_exposure_rate": _safe_rate(repeat, interventions),
        "interruption_rate": _safe_rate(interruptions, interventions),
        "grounded_recommendation_rate": _safe_rate(grounded, interventions),
        "watch_rate": _safe_rate(watch_count, watch_count + skip_count),
        "skip_rate": _safe_rate(skip_count, watch_count + skip_count),
        "like_rate": _safe_rate(like_count, watch_count + skip_count),
        "mean_dwell_seconds": _safe_rate(
            dwell_seconds, watch_count + skip_count
        ),
        "complementary_bundle_precision": _safe_rate(
            complementary_bundle_count, bundle_offer_count
        ),
        "unsupported_claim_rate": _safe_rate(len(unsupported_indices), interventions),
        "mean_tool_calls_per_step": _safe_rate(tool_calls, steps),
        "net_revenue": net_revenue,
        "fatigue": fatigue,
        "terminal_fatigue": fatigue,
        "terminal_satisfaction": terminal_satisfaction,
        "long_term_return": total_reward,
    }
    # Concise aliases are useful in CSV reports while canonical names remain explicit.
    metrics.update(
        {
            "correct_no_rec_rate": metrics["correct_no_recommend_rate"],
            "irrelevant_rate": metrics["irrelevant_recommendation_rate"],
            "repeat_rate": metrics["repeat_exposure_rate"],
            "grounded_rate": metrics["grounded_recommendation_rate"],
        }
    )
    return metrics


def aggregate_evaluations(evaluations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro-average count-derived rates and macro-average episode outcomes."""

    rows = [dict(row) for row in evaluations]
    if not rows:
        empty = evaluate_episode([])
        empty["episodes"] = 0
        empty["per_episode"] = []
        return empty

    count_fields = (
        "steps",
        "interventions",
        "no_recommendations",
        "clicks",
        "add_to_carts",
        "watches",
        "skips",
        "likes",
        "purchases",
        "qualified_purchases",
        "returns",
        "correct_no_recommendations",
        "irrelevant_recommendations",
        "repeat_exposures",
        "interruptions",
        "grounded_recommendations",
        "unsupported_claims",
        "bundle_offers",
        "complementary_bundle_offers",
        "tool_calls",
    )
    totals = {
        field: sum(int(_number(row.get(field))) for row in rows) for field in count_fields
    }
    interventions = totals["interventions"]
    purchases = totals["purchases"]
    clicks = totals["clicks"]
    carts = totals["add_to_carts"]
    content_views = totals["watches"] + totals["skips"]
    total_dwell_seconds = math.fsum(
        _number(row.get("total_dwell_seconds")) for row in rows
    )
    no_recommendations = totals["no_recommendations"]
    result: dict[str, Any] = {
        "episodes": len(rows),
        **totals,
        "click_rate": _safe_rate(clicks, interventions),
        "add_to_cart_rate": _safe_rate(carts, interventions),
        "purchase_rate": _safe_rate(purchases, interventions),
        "click_to_cart_rate": _safe_rate(carts, clicks),
        "cart_to_purchase_rate": _safe_rate(purchases, carts),
        "qualified_purchase_rate": _safe_rate(totals["qualified_purchases"], purchases),
        "correct_no_recommend_rate": _safe_rate(
            totals["correct_no_recommendations"], no_recommendations
        ),
        "interventions_per_100": _safe_rate(100.0 * interventions, totals["steps"]),
        "return_rate": _safe_rate(totals["returns"], purchases),
        "irrelevant_recommendation_rate": _safe_rate(
            totals["irrelevant_recommendations"], interventions
        ),
        "repeat_exposure_rate": _safe_rate(totals["repeat_exposures"], interventions),
        "interruption_rate": _safe_rate(totals["interruptions"], interventions),
        "grounded_recommendation_rate": _safe_rate(
            totals["grounded_recommendations"], interventions
        ),
        "total_dwell_seconds": total_dwell_seconds,
        "watch_rate": _safe_rate(totals["watches"], content_views),
        "skip_rate": _safe_rate(totals["skips"], content_views),
        "like_rate": _safe_rate(totals["likes"], content_views),
        "mean_dwell_seconds": _safe_rate(total_dwell_seconds, content_views),
        "complementary_bundle_precision": _safe_rate(
            totals["complementary_bundle_offers"], totals["bundle_offers"]
        ),
        "unsupported_claim_rate": _safe_rate(
            totals["unsupported_claims"], interventions
        ),
        "mean_tool_calls_per_step": _safe_rate(totals["tool_calls"], totals["steps"]),
        "net_revenue": math.fsum(_number(row.get("net_revenue")) for row in rows),
        "fatigue": math.fsum(_number(row.get("fatigue")) for row in rows) / len(rows),
        "terminal_fatigue": math.fsum(
            _number(row.get("terminal_fatigue")) for row in rows
        )
        / len(rows),
        "terminal_satisfaction": math.fsum(
            _number(row.get("terminal_satisfaction")) for row in rows
        )
        / len(rows),
        "mean_terminal_satisfaction": math.fsum(
            _number(row.get("terminal_satisfaction")) for row in rows
        )
        / len(rows),
        "long_term_return": math.fsum(
            _number(row.get("long_term_return")) for row in rows
        )
        / len(rows),
        "per_episode": rows,
    }
    result.update(
        {
            "correct_no_rec_rate": result["correct_no_recommend_rate"],
            "irrelevant_rate": result["irrelevant_recommendation_rate"],
            "repeat_rate": result["repeat_exposure_rate"],
            "grounded_rate": result["grounded_recommendation_rate"],
        }
    )
    return result


def evaluate_episodes(episodes: Iterable[Any]) -> dict[str, Any]:
    """Evaluate episodes individually and return a comparison-ready aggregate."""

    return aggregate_evaluations(evaluate_episode(episode) for episode in episodes)


# Stable short aliases for command-line/report code.
compute_feed_metrics = evaluate_episode
evaluate = evaluate_episode


__all__ = [
    "aggregate_evaluations",
    "compute_feed_metrics",
    "evaluate",
    "evaluate_episode",
    "evaluate_episodes",
]
