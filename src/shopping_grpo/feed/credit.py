"""Long-horizon return, event attribution, and CRN counterfactual credit.

Counterfactual replay uses a caller-supplied environment factory.  Each call to the
factory must construct the same episode seed and calibration, so factual and
counterfactual trajectories consume the simulator's action-invariant random-number
streams.  Replay never reads environment internals: recorded information calls are
reissued, evidence is intersected with the current public state, and an action that
cannot be safely rebuilt degrades to ``no_recommend``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from shopping_grpo.feed.evidence import has_product_evidence
from shopping_grpo.feed.policies import transition_from_step_result
from shopping_grpo.feed.schema import EpisodeResult, FeedTransition


def _number(value: Any, *, name: str = "value") -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _gamma(value: Any) -> float:
    gamma = _number(value, name="gamma")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    return gamma


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


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return []


def _trajectory(value: Any) -> list[Any]:
    if isinstance(value, EpisodeResult):
        return list(value.transitions)
    if isinstance(value, Mapping):
        transitions = value.get("transitions", value.get("trajectory"))
        if isinstance(transitions, Sequence) and not isinstance(transitions, (str, bytes)):
            return list(transitions)
    if hasattr(value, "transitions"):
        transitions = getattr(value, "transitions")
        if isinstance(transitions, Sequence):
            return list(transitions)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError("trajectory must be an EpisodeResult or a sequence")


def _transition_row(value: Any) -> dict[str, Any]:
    return _row(value)


def _events(transition: Any) -> list[dict[str, Any]]:
    row = _transition_row(transition)
    metadata = _row(row.get("metadata"))
    raw = metadata.get("raw_events")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = row.get("events", row.get("user_events", ()))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [_row(event) for event in raw]


def _reward_breakdown(transition: Any) -> dict[str, Any]:
    row = _transition_row(transition)
    metadata = _row(row.get("metadata"))
    raw = metadata.get("raw_reward_breakdown")
    if isinstance(raw, Mapping):
        return dict(raw)
    return _row(row.get("reward"))


def _reward_total(transition: Any) -> float:
    reward = _reward_breakdown(transition)
    if "total" in reward:
        return _number(reward["total"], name="reward total")
    if isinstance(transition, FeedTransition):
        return transition.reward.total
    return math.fsum(
        _number(value, name=f"reward.{key}")
        for key, value in reward.items()
        if key != "total"
    )


def discounted_returns(
    rewards: Iterable[float],
    gamma: float = 1.0,
    *,
    bootstrap_value: float = 0.0,
) -> list[float]:
    """Return the scalar reward-to-go at every step in linear time."""

    discount = _gamma(gamma)
    values = [_number(value, name="reward") for value in rewards]
    running = _number(bootstrap_value, name="bootstrap_value")
    returns = [0.0] * len(values)
    for index in range(len(values) - 1, -1, -1):
        running = values[index] + discount * running
        returns[index] = running
    return returns


def _event_name(event: Mapping[str, Any]) -> str:
    return str(event.get("event_type", event.get("type", event.get("event", ""))))


def _event_step(event: Mapping[str, Any], default: int = 0) -> int:
    raw = event.get("step", default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return default
    return raw


def _source_step(event: Mapping[str, Any], default: int) -> int:
    metadata = _row(event.get("metadata"))
    raw = event.get("source_step", metadata.get("source_step", default))
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return default
    return raw


def _event_weight(
    event: Mapping[str, Any], event_rewards: Mapping[str, float] | None
) -> float | None:
    for field in ("credit", "reward", "reward_value"):
        if field in event:
            return _number(event[field], name=f"event.{field}")
    if event_rewards is not None and _event_name(event) in event_rewards:
        count = event.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            count = 1
        return _number(event_rewards[_event_name(event)], name="event reward") * count
    return None


def _looks_like_events(items: Sequence[Any]) -> bool:
    if not items:
        return True
    first = _row(items[0])
    return "action" not in first and "events" not in first and bool(
        {"event_type", "event", "type"} & set(first)
    )


def _explicit_purchase_credit(event: Mapping[str, Any]) -> float | None:
    metadata = _row(event.get("metadata"))
    fields = ("qualified_purchase_credit", "satisfaction_credit")
    if not any(field in metadata for field in fields):
        return None
    return math.fsum(
        _number(metadata.get(field, 0.0), name=f"event.metadata.{field}")
        for field in fields
    )


def event_credit_by_source(
    data: Any,
    event_rewards: Mapping[str, float] | None = None,
    *,
    gamma: float = 1.0,
) -> dict[int, float]:
    """Attribute delayed event value to the action recorded in ``source_step``.

    For a raw event sequence, explicit ``credit``/``reward`` values (or the supplied
    ``event_rewards`` table) are grouped directly.  For a trajectory, the full step
    reward is retained while purchase and return components are moved from their
    realization step to the originating recommendation.  This preserves total credit
    when ``gamma=1`` and prevents delayed purchases from being assigned to an unrelated
    later action.
    """

    discount = _gamma(gamma)
    if event_rewards is not None:
        event_rewards = {
            str(key): _number(value, name=f"event_rewards[{key!r}]")
            for key, value in event_rewards.items()
        }
    items = _trajectory(data) if not (
        isinstance(data, Sequence)
        and not isinstance(data, (str, bytes))
        and _looks_like_events(data)
    ) else list(data)

    if _looks_like_events(items):
        credits: dict[int, float] = {}
        for raw_event in items:
            event = _row(raw_event)
            realized = _event_step(event)
            source = _source_step(event, realized)
            value = _event_weight(event, event_rewards)
            if value is None:
                continue
            value *= discount ** max(realized - source, 0)
            credits[source] = credits.get(source, 0.0) + value
        return dict(sorted(credits.items()))

    if event_rewards is not None:
        raw_events = [event for transition in items for event in _events(transition)]
        return event_credit_by_source(raw_events, event_rewards, gamma=discount)

    credits: dict[int, float] = {}
    for position, transition in enumerate(items):
        transition_row = _transition_row(transition)
        step = transition_row.get("step", position)
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            step = position
        credits[step] = credits.get(step, 0.0) + _reward_total(transition)

        events = _events(transition)
        delayed_purchases = [
            event
            for event in events
            if _event_name(event) == "purchase" and _source_step(event, step) != step
        ]
        delayed_returns = [
            event
            for event in events
            if _event_name(event) == "return" and _source_step(event, step) != step
        ]
        reward = _reward_breakdown(transition)
        purchase_credit = math.fsum(
            _number(reward.get(name, 0.0), name=f"reward.{name}")
            for name in (
                "qualified_purchase_value",
                "qualified_purchase",
                "revenue",
                "purchase_satisfaction",
            )
        )
        return_credit = _number(reward.get("return_penalty", 0.0), name="return_penalty")
        if delayed_purchases:
            explicit = [_explicit_purchase_credit(event) for event in delayed_purchases]
            purchase_allocations = (
                [float(value) for value in explicit if value is not None]
                if all(value is not None for value in explicit)
                else [purchase_credit / len(delayed_purchases)] * len(delayed_purchases)
            )
            credits[step] -= math.fsum(purchase_allocations)
            for event, amount in zip(delayed_purchases, purchase_allocations):
                source = _source_step(event, step)
                realized = _event_step(event, step)
                attributed = amount * discount ** max(realized - source, 0)
                credits[source] = credits.get(source, 0.0) + attributed
        if delayed_returns and return_credit != 0.0:
            credits[step] -= return_credit
            share = return_credit / len(delayed_returns)
            for event in delayed_returns:
                source = _source_step(event, step)
                realized = _event_step(event, step)
                attributed = share * discount ** max(realized - source, 0)
                credits[source] = credits.get(source, 0.0) + attributed
    return dict(sorted(credits.items()))


def _action(transition: Any) -> dict[str, Any]:
    return _row(_transition_row(transition).get("action"))


def _tool_records(transition: Any) -> list[dict[str, Any]]:
    row = _transition_row(transition)
    raw = row.get("tool_records", row.get("tools", ()))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [_row(record) for record in raw]


def _no_recommend() -> dict[str, Any]:
    return {
        "decision": "no_recommend",
        "surface": "none",
        "strategy": "none",
        "product_ids": [],
        "evidence_ids": [],
        "explanation": "CRN counterfactual no-intervention baseline.",
    }


def _query(observation: Mapping[str, Any]) -> str:
    video = _row(observation.get("current_video"))
    terms: list[str] = []
    for field in ("caption", "scene", "scenes", "objects", "style", "topics", "asr", "ocr"):
        terms.extend(_strings(video.get(field)))
    return " ".join(dict.fromkeys(terms)) or "shopping products"


def _remaining_calls(observation: Mapping[str, Any]) -> int:
    maximum = observation.get("max_info_tool_calls", 3)
    used = observation.get("info_tool_calls", 0)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        maximum = 3
    if isinstance(used, bool) or not isinstance(used, int):
        used = 0
    return max(maximum - used, 0)


def _replay_tools(env: Any, records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        name = str(record.get("tool_name", record.get("name", "")))
        if not name or name == "commit_recommendation":
            continue
        if _remaining_calls(env.observation()) <= 0:
            return
        arguments = _row(record.get("arguments", record.get("parameters")))
        try:
            env.call_tool(name, arguments)
        except (KeyError, RuntimeError, TypeError, ValueError):
            # A previous counterfactual can change purchased/visible products.  The
            # action re-builder below makes one last public retrieval attempt.
            continue


def _rebuild_action(env: Any, original: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(original.get("decision", ""))
    if decision == "no_recommend":
        action = _no_recommend()
        action["explanation"] = str(
            original.get("explanation", original.get("reason", action["explanation"]))
        )
        return action
    if decision == "delay":
        return {
            "decision": "delay",
            "surface": "none",
            "strategy": "none",
            "product_ids": [],
            "evidence_ids": [],
            "explanation": str(original.get("explanation", "Replayed delay action.")),
        }
    if decision != "recommend":
        return _no_recommend()

    state = env.observation()
    requested = _strings(original.get("product_ids"))[:2]
    visible = set(_strings(state.get("visible_product_ids")))
    purchased = set(_strings(state.get("purchased_product_ids")))
    if not requested or not set(requested).issubset(visible - purchased):
        if _remaining_calls(state) > 0:
            try:
                env.call_tool("retrieve_products", {"query": _query(state)})
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        state = env.observation()
        visible = set(_strings(state.get("visible_product_ids")))
        purchased = set(_strings(state.get("purchased_product_ids")))
    selected = [product_id for product_id in requested if product_id in visible - purchased]
    if not selected:
        return _no_recommend()

    available = set(_strings(state.get("evidence_ids")))
    original_evidence = [
        identifier
        for identifier in _strings(original.get("evidence_ids"))
        if identifier in available
    ]
    context = [
        identifier
        for identifier in available
        if identifier.startswith(("video.", "history.", "persona."))
    ]
    product = [
        identifier
        for identifier in available
        if any(has_product_evidence([identifier], item) for item in selected)
    ]
    if not product and _remaining_calls(state) > 0:
        try:
            env.call_tool("inspect_product", {"product_id": selected[0]})
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
        state = env.observation()
        available = set(_strings(state.get("evidence_ids")))
        product = [
            identifier
            for identifier in available
            if any(has_product_evidence([identifier], item) for item in selected)
        ]
    if not context or not product:
        return _no_recommend()
    evidence = list(dict.fromkeys((*original_evidence, *sorted(context)[:3], *sorted(product)[:8])))
    return {
        "decision": "recommend",
        "surface": str(original.get("surface", "product_card")),
        "strategy": str(original.get("strategy", "direct")),
        "relationship": str(original.get("relationship", "primary")),
        "product_ids": selected,
        "evidence_ids": evidence,
        "explanation": str(
            original.get("explanation", original.get("reason", "Replayed grounded action."))
        )[:500],
    }


def _step_observation(result: Any) -> dict[str, Any]:
    observation = getattr(result, "observation", None)
    if observation is None:
        observation = _row(result).get("observation")
    return _row(observation)


def _replay_on_env(
    env: Any,
    trajectory: Any,
    *,
    replace_step: int | None,
    replacement_action: Mapping[str, Any] | None = None,
) -> EpisodeResult:
    source = _trajectory(trajectory)
    initial = env.observation()
    episode_id = str(initial.get("episode_id", "episode"))
    state = initial
    transitions: list[FeedTransition] = []
    for position, recorded in enumerate(source):
        if state.get("done"):
            break
        recorded_row = _transition_row(recorded)
        step = state.get("step", position)
        if isinstance(step, bool) or not isinstance(step, int):
            step = position
        if replace_step != step:
            _replay_tools(env, _tool_records(recorded))
            action = _rebuild_action(env, _action(recorded))
        else:
            action = (
                _no_recommend()
                if replacement_action is None
                else dict(replacement_action)
            )
        pre_observation = state
        try:
            result = env.step(action)
        except ValueError:
            action = _no_recommend()
            result = env.step(action)
        transitions.append(
            transition_from_step_result(
                step=step,
                observation=pre_observation,
                action=action,
                step_result=result,
            )
        )
        state = _step_observation(result)

    # A partial factual record still receives a complete long-term baseline.
    while not state.get("done"):
        step = state.get("step", len(transitions))
        if isinstance(step, bool) or not isinstance(step, int):
            step = len(transitions)
        pre_observation = state
        action = _no_recommend()
        result = env.step(action)
        transitions.append(
            transition_from_step_result(
                step=step,
                observation=pre_observation,
                action=action,
                step_result=result,
            )
        )
        state = _step_observation(result)

    final_state = dict(state)
    try:
        summary = env.summary()
    except AttributeError:
        summary = None
    if isinstance(summary, Mapping):
        final_state["evaluation_summary"] = dict(summary)
    return EpisodeResult(
        episode_id=episode_id,
        transitions=tuple(transitions),
        done=bool(state.get("done")),
        termination_reason="feed_exhausted" if state.get("done") else "partial_replay",
        final_state=final_state,
        metadata={
            "replay": "common_random_numbers",
            "replaced_step": replace_step,
        },
    )


def replay_episode(
    env_factory: Callable[[], Any],
    trajectory: Any,
    *,
    replace_step: int | None = None,
    replacement_action: Mapping[str, Any] | None = None,
) -> EpisodeResult:
    """Replay actions, optionally replacing one with an exact non-marketing action."""

    if not callable(env_factory):
        raise TypeError("env_factory must be callable")
    if replace_step is not None and (
        isinstance(replace_step, bool) or not isinstance(replace_step, int) or replace_step < 0
    ):
        raise ValueError("replace_step must be a non-negative integer or None")
    _validate_replacement_action(replacement_action, replace_step=replace_step)
    return _replay_on_env(
        env_factory(),
        trajectory,
        replace_step=replace_step,
        replacement_action=replacement_action,
    )


def _validate_replacement_action(
    replacement_action: Mapping[str, Any] | None,
    *,
    replace_step: int | None,
) -> None:
    if replacement_action is None:
        return
    if replace_step is None:
        raise ValueError("replacement_action requires replace_step")
    if not isinstance(replacement_action, Mapping):
        raise TypeError("replacement_action must be a mapping")
    if str(replacement_action.get("decision")) not in {"delay", "no_recommend"}:
        raise ValueError("replacement_action must be delay or no_recommend")


def _initial_signature(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    video = _row(observation.get("current_video"))
    return (
        observation.get("episode_id"),
        observation.get("total_steps"),
        observation.get("step"),
        video.get("video_id"),
    )


def _long_term_return(result: EpisodeResult, gamma: float) -> float:
    returns = discounted_returns(
        (transition.reward.total for transition in result.transitions), gamma
    )
    return returns[0] if returns else 0.0


def counterfactual_advantage(
    env_factory: Callable[[], Any],
    trajectory: Any,
    step: int,
    *,
    gamma: float = 1.0,
    replacement_action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate ``A_cf`` by CRN replay with one exact non-marketing replacement."""

    if not callable(env_factory):
        raise TypeError("env_factory must be callable")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    _validate_replacement_action(replacement_action, replace_step=step)
    discount = _gamma(gamma)
    factual_env = env_factory()
    counterfactual_env = env_factory()
    factual_initial = factual_env.observation()
    counterfactual_initial = counterfactual_env.observation()
    if _initial_signature(factual_initial) != _initial_signature(counterfactual_initial):
        raise ValueError("env_factory must reproduce the same episode for CRN replay")

    factual = _replay_on_env(factual_env, trajectory, replace_step=None)
    counterfactual = _replay_on_env(
        counterfactual_env,
        trajectory,
        replace_step=step,
        replacement_action=replacement_action,
    )
    factual_return = _long_term_return(factual, discount)
    counterfactual_return = _long_term_return(counterfactual, discount)
    return {
        "step": step,
        "method": "common_random_numbers",
        "factual_return": factual_return,
        "counterfactual_return": counterfactual_return,
        "counterfactual_action": (
            _no_recommend()
            if replacement_action is None
            else dict(replacement_action)
        ),
        "A_cf": factual_return - counterfactual_return,
        "factual_episode": factual.to_dict(),
        "counterfactual_episode": counterfactual.to_dict(),
    }


def counterfactual_advantages(
    env_factory: Callable[[], Any],
    trajectory: Any,
    *,
    steps: Iterable[int] | None = None,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Compute CRN advantages for selected (by default recommended) decisions."""

    source = _trajectory(trajectory)
    if steps is None:
        selected_steps = [
            int(_transition_row(item).get("step", index))
            for index, item in enumerate(source)
            if str(_action(item).get("decision")) == "recommend"
        ]
    else:
        selected_steps = list(steps)
    for step in selected_steps:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("steps must contain non-negative integers")
    selected_steps = list(dict.fromkeys(selected_steps))
    rows = [
        counterfactual_advantage(
            env_factory,
            source,
            step,
            gamma=gamma,
        )
        for step in selected_steps
    ]
    return {
        "method": "common_random_numbers",
        "advantages": rows,
        "A_cf": {row["step"]: row["A_cf"] for row in rows},
    }


# Explicit names make the experimental method easy to discover from configs.
common_random_number_counterfactual = counterfactual_advantage
crn_counterfactual_advantage = counterfactual_advantage


__all__ = [
    "common_random_number_counterfactual",
    "counterfactual_advantage",
    "counterfactual_advantages",
    "crn_counterfactual_advantage",
    "discounted_returns",
    "event_credit_by_source",
    "replay_episode",
]
