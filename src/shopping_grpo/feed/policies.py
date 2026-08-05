"""Public-state baseline policies and rollout helpers for the Feed profile.

The policies in this module deliberately know nothing about simulator internals or
the product catalog object.  They observe the same state as a model policy, obtain
product facts through information tools, and return one guarded When--What--How
action.  This makes the baselines suitable for behavior-log generation as well as
fair evaluation against a trained agent.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shopping_grpo.feed.evidence import has_product_evidence
from shopping_grpo.feed.schema import (
    EpisodeResult,
    EventType,
    FeedAction,
    FeedTransition,
    RewardBreakdown,
    ToolRecord,
    UserEvent,
)


Action = dict[str, Any]

_CONTEXT_EVIDENCE_PREFIXES = ("video.", "history.", "persona.")
_RECOMMENDATION_SURFACES = {
    "product_card",
    "coupon",
    "review_summary",
    "price_comparison",
    "similar_products",
    "bundle",
    "creator_video",
}
_RECOMMENDATION_STRATEGIES = {
    "direct",
    "review_summary",
    "price_comparison",
    "discount",
    "cheaper_alternative",
    "similar",
    "complement",
    "bundle",
}


def _mapping(value: Any) -> dict[str, Any]:
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
    return [str(value)] if str(value) else []


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _event_type(event: Any) -> str:
    row = _mapping(event)
    return str(row.get("event_type", row.get("type", row.get("event", ""))))


def _no_recommend(
    explanation: str = "No sufficiently grounded intervention is available.",
) -> Action:
    return {
        "decision": "no_recommend",
        "surface": "none",
        "strategy": "none",
        "product_ids": [],
        "evidence_ids": [],
        "explanation": explanation,
    }


def _delay(explanation: str = "Delay the intervention and observe more public feedback.") -> Action:
    return {
        "decision": "delay",
        "surface": "none",
        "strategy": "none",
        "product_ids": [],
        "evidence_ids": [],
        "explanation": explanation,
    }


def _current_query(observation: Mapping[str, Any]) -> str:
    video = _mapping(observation.get("current_video"))
    persona = _mapping(observation.get("persona"))
    terms: list[str] = []
    for field in ("caption", "scene", "scenes", "objects", "style", "topics", "asr", "ocr"):
        terms.extend(_strings(video.get(field)))
    for field in ("category_interests", "style_preferences"):
        value = persona.get(field)
        if isinstance(value, Mapping):
            ranked = sorted(
                value.items(),
                key=lambda item: (-_number(item[1]), str(item[0])),
            )
            terms.extend(str(key) for key, _ in ranked[:4])
        else:
            terms.extend(_strings(value))
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = " ".join(str(term).split())
        if term and term.casefold() not in seen:
            normalized.append(term)
            seen.add(term.casefold())
    return " ".join(normalized[:16]) or "shopping products"


def _remaining_calls(observation: Mapping[str, Any]) -> int:
    maximum = observation.get("max_info_tool_calls", 3)
    used = observation.get("info_tool_calls", 0)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        maximum = 3
    if isinstance(used, bool) or not isinstance(used, int):
        used = 0
    return max(maximum - used, 0)


def _context_evidence(observation: Mapping[str, Any]) -> list[str]:
    evidence = _strings(observation.get("evidence_ids"))
    contextual = [
        identifier
        for identifier in evidence
        if identifier.startswith(_CONTEXT_EVIDENCE_PREFIXES)
    ]
    return contextual[:3]


def _evidence_mentions_product(identifier: str, product_id: str) -> bool:
    return has_product_evidence([identifier], product_id)


def _product_evidence(
    observation: Mapping[str, Any], product_ids: Sequence[str]
) -> list[str]:
    evidence = _strings(observation.get("evidence_ids"))
    return [
        identifier
        for identifier in evidence
        if identifier.startswith("product.")
        and any(
            _evidence_mentions_product(identifier, product_id)
            for product_id in product_ids
        )
    ][:8]


def _candidate_rows(result: Any) -> list[dict[str, Any]]:
    payload = _mapping(result)
    for key in ("products", "results", "items", "alternatives", "complements"):
        raw_rows = payload.get(key)
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
            return [row for item in raw_rows if (row := _mapping(item))]
    return []


def _eligible_rows(
    rows: Sequence[Mapping[str, Any]], observation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    purchased = set(_strings(observation.get("purchased_product_ids")))
    eligible: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        product_id = str(row.get("product_id", row.get("asin", row.get("id", ""))))
        if not product_id or product_id in purchased:
            continue
        stock = row.get("stock", row.get("inventory"))
        if stock is not None and _number(stock, -1.0) <= 0.0:
            continue
        row["product_id"] = product_id
        eligible.append(row)
    return eligible


def _retrieve(env: Any, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    if _remaining_calls(observation) <= 0:
        visible = _strings(observation.get("visible_product_ids"))
        return _eligible_rows(
            [{"product_id": product_id} for product_id in visible], observation
        )
    try:
        result = env.call_tool("retrieve_products", {"query": _current_query(observation)})
    except (KeyError, RuntimeError, TypeError, ValueError):
        return []
    current = env.observation()
    return _eligible_rows(_candidate_rows(result), current)


def _inspect(env: Any, product_id: str) -> dict[str, Any]:
    observation = env.observation()
    if _remaining_calls(observation) <= 0:
        return {}
    try:
        result = env.call_tool("inspect_product", {"product_id": product_id})
    except (KeyError, RuntimeError, TypeError, ValueError):
        return {}
    payload = _mapping(result)
    return _mapping(payload.get("product"))


def _read_reviews(env: Any, product_id: str) -> dict[str, Any]:
    observation = env.observation()
    if _remaining_calls(observation) <= 0:
        return {}
    try:
        return _mapping(env.call_tool("read_reviews", {"product_id": product_id}))
    except (KeyError, RuntimeError, TypeError, ValueError):
        return {}


def _grounded_recommendation(
    env: Any,
    product_ids: Sequence[str],
    *,
    surface: str,
    strategy: str,
    relationship: str = "primary",
    explanation: str,
) -> Action:
    state = env.observation()
    visible = set(_strings(state.get("visible_product_ids")))
    purchased = set(_strings(state.get("purchased_product_ids")))
    selected = [
        str(product_id)
        for product_id in product_ids
        if str(product_id) in visible and str(product_id) not in purchased
    ][:2]
    if not selected:
        return _no_recommend("No eligible visible product remains.")

    product_evidence = _product_evidence(state, selected)
    for product_id in selected:
        if any(
            _evidence_mentions_product(identifier, product_id)
            for identifier in product_evidence
        ):
            continue
        if _remaining_calls(state) <= 0:
            break
        _inspect(env, product_id)
        state = env.observation()
        product_evidence = _product_evidence(state, selected)
    context_evidence = _context_evidence(state)
    all_products_grounded = all(
        any(
            _evidence_mentions_product(identifier, product_id)
            for identifier in product_evidence
        )
        for product_id in selected
    )
    if not all_products_grounded or not context_evidence:
        return _no_recommend("The recommendation lacks visible product or context evidence.")
    if surface not in _RECOMMENDATION_SURFACES:
        surface = "product_card"
    if strategy not in _RECOMMENDATION_STRATEGIES:
        strategy = "direct"
    if relationship not in {"primary", "alternative", "complement", "bundle"}:
        relationship = "primary"
    return {
        "decision": "recommend",
        "surface": surface,
        "strategy": strategy,
        "relationship": relationship,
        "product_ids": selected,
        "evidence_ids": list(dict.fromkeys((*context_evidence, *product_evidence))),
        "explanation": explanation[:500],
    }


class FeedPolicy:
    """Small common interface shared by heuristic and model-backed policies."""

    name = "feed_policy"

    def act(self, env: Any) -> Action:
        raise NotImplementedError

    def select_action(self, env: Any) -> Action:
        return self.act(env)

    def decide(self, env: Any) -> Action:
        return self.act(env)

    def __call__(self, env: Any) -> Action:
        return self.act(env)


class RandomPolicy(FeedPolicy):
    """Deterministic-per-state random behavior policy for coverage collection."""

    name = "random"

    def __init__(
        self,
        seed: int = 0,
        *,
        recommendation_probability: float = 0.55,
        delay_probability: float = 0.20,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        for name, value in (
            ("recommendation_probability", recommendation_probability),
            ("delay_probability", delay_probability),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if recommendation_probability + delay_probability > 1.0:
            raise ValueError("recommendation_probability + delay_probability must not exceed 1")
        self.seed = seed
        self.recommendation_probability = float(recommendation_probability)
        self.delay_probability = float(delay_probability)

    def _rng(self, observation: Mapping[str, Any]) -> random.Random:
        material = (
            f"{self.seed}|{observation.get('episode_id', '')}|"
            f"{observation.get('step', 0)}"
        ).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(derived_seed)

    def act(self, env: Any) -> Action:
        observation = env.observation()
        if observation.get("done"):
            raise RuntimeError("cannot act on a terminal feed episode")
        rng = self._rng(observation)
        draw = rng.random()
        if draw >= self.recommendation_probability:
            if draw < self.recommendation_probability + self.delay_probability:
                return _delay("Random coverage policy delays this intervention.")
            return _no_recommend("Random coverage policy suppresses this intervention.")

        rows = _retrieve(env, observation)
        if not rows:
            return _no_recommend("Random policy found no eligible public candidate.")
        selected = rows[rng.randrange(len(rows))]
        surface, strategy = rng.choice(
            (
                ("product_card", "direct"),
                ("coupon", "discount"),
                ("review_summary", "review_summary"),
                ("similar_products", "similar"),
            )
        )
        return _grounded_recommendation(
            env,
            [selected["product_id"]],
            surface=surface,
            strategy=strategy,
            explanation=(
                "Randomized behavior action grounded in the current video and retrieved product."
            ),
        )


class PopularPolicy(FeedPolicy):
    """Recommend the strongest public popularity/rating candidate for the video."""

    name = "popular"

    def act(self, env: Any) -> Action:
        observation = env.observation()
        rows = _retrieve(env, observation)
        if not rows:
            return _no_recommend("No popular eligible product was retrieved.")
        selected = max(
            rows,
            key=lambda row: (
                _number(row.get("popularity")),
                _number(row.get("rating")),
                _number(row.get("stock", row.get("inventory"))),
                -_number(row.get("price")),
                str(row.get("product_id")),
            ),
        )
        return _grounded_recommendation(
            env,
            [selected["product_id"]],
            surface="product_card",
            strategy="direct",
            explanation=(
                "Selected from public retrieval results using popularity, rating, and availability."
            ),
        )


class SimilarityPolicy(FeedPolicy):
    """Use the information tool's content-aware ranking without private state."""

    name = "similarity"

    def act(self, env: Any) -> Action:
        observation = env.observation()
        rows = _retrieve(env, observation)
        if not rows:
            return _no_recommend("No content-similar eligible product was retrieved.")
        # Retrieval order is the public similarity rank.  A public score, when
        # supplied by another conforming environment, takes precedence.
        ranked = sorted(
            enumerate(rows),
            key=lambda item: (
                -_number(item[1].get("similarity", item[1].get("score"))),
                item[0],
            ),
        )
        selected = ranked[0][1]
        return _grounded_recommendation(
            env,
            [selected["product_id"]],
            surface="similar_products",
            strategy="similar",
            explanation=(
                "The retrieved product is most similar to the current public video context."
            ),
        )


def _recent_interventions(observation: Mapping[str, Any]) -> int:
    return sum(
        1
        for event in observation.get("recent_events", ())
        if _event_type(event) in {"intervention", "repeat_exposure"}
    )


def _public_fit_score(
    row: Mapping[str, Any], observation: Mapping[str, Any]
) -> float:
    video = _mapping(observation.get("current_video"))
    persona = _mapping(observation.get("persona"))
    video_terms = {
        term.casefold()
        for field in ("caption", "scene", "scenes", "objects", "style", "topics")
        for term in _strings(video.get(field))
    }
    product_terms = {
        term.casefold()
        for field in ("title", "category", "attributes", "tags")
        for term in _strings(row.get(field))
    }
    overlap = len(video_terms & product_terms) / max(len(video_terms), 1)
    budget = _number(persona.get("budget"), 0.0)
    price = _number(row.get("price"), 0.0)
    price_fit = 1.0 if budget <= 0.0 else max(0.0, 1.0 - price / max(budget, 1.0))
    return 0.50 * overlap + 0.25 * (_number(row.get("rating"), 0.0) / 5.0) + 0.25 * price_fit


class RulePolicy(FeedPolicy):
    """Experience-aware baseline using only observable fatigue proxies and facts."""

    name = "rule"

    def __init__(self, *, max_recent_interventions: int = 3) -> None:
        if isinstance(max_recent_interventions, bool) or not isinstance(
            max_recent_interventions, int
        ):
            raise TypeError("max_recent_interventions must be an integer")
        if max_recent_interventions < 1:
            raise ValueError("max_recent_interventions must be positive")
        self.max_recent_interventions = max_recent_interventions

    def act(self, env: Any) -> Action:
        observation = env.observation()
        if _recent_interventions(observation) >= self.max_recent_interventions:
            return _delay("Recent public history shows frequent commercial interventions.")
        rows = _retrieve(env, observation)
        if not rows:
            return _no_recommend("No eligible rule-matching product was retrieved.")
        selected = max(
            rows,
            key=lambda row: (_public_fit_score(row, observation), str(row["product_id"])),
        )
        details = _inspect(env, selected["product_id"])
        if details:
            selected = {**selected, **details}

        persona = _mapping(observation.get("persona"))
        budget = _number(persona.get("budget"), 0.0)
        price = _number(selected.get("price"), 0.0)
        price_sensitive = _number(persona.get("price_sensitivity"), 0.5) >= 0.65
        risk_tags = _strings(selected.get("risk_tags", selected.get("tags")))
        if _remaining_calls(env.observation()) > 0:
            reviews = _read_reviews(env, selected["product_id"])
            risk_tags.extend(_strings(reviews.get("risk_tags")))

        if budget > 0.0 and price > budget:
            return _no_recommend("The retrieved product exceeds the public persona budget.")
        if risk_tags:
            surface, strategy = "review_summary", "review_summary"
            explanation = (
                "The item fits the public context and its review risks are explicitly disclosed."
            )
        elif price_sensitive:
            surface, strategy = "coupon", "discount"
            explanation = (
                "The item fits the public context; a discount addresses stated price sensitivity."
            )
        else:
            surface, strategy = "product_card", "direct"
            explanation = (
                "Public video, product, price, and persona evidence support this intervention."
            )
        return _grounded_recommendation(
            env,
            [selected["product_id"]],
            surface=surface,
            strategy=strategy,
            explanation=explanation,
        )


class _TeacherContext:
    """Narrow facade supplied to an optional external teacher callback."""

    __slots__ = ("__observation", "__call_tool")

    def __init__(self, observation: Callable[[], Any], call_tool: Callable[..., Any]) -> None:
        self.__observation = observation
        self.__call_tool = call_tool

    def observation(self) -> Any:
        return self.__observation()

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        return self.__call_tool(name, arguments or {})


def _normalize_teacher_action(value: Any, observation: Mapping[str, Any]) -> Action | None:
    action = _mapping(value)
    decision = str(action.get("decision", ""))
    if decision in {"delay", "no_recommend"}:
        if decision == "delay":
            return _delay(
                str(action.get("explanation", "Teacher delayed the intervention."))
            )
        return _no_recommend(
            str(action.get("explanation", "Teacher suppressed the intervention."))
        )
    if decision != "recommend":
        return None
    visible = set(_strings(observation.get("visible_product_ids")))
    purchased = set(_strings(observation.get("purchased_product_ids")))
    evidence = set(_strings(observation.get("evidence_ids")))
    product_ids = _strings(action.get("product_ids"))[:2]
    evidence_ids = _strings(action.get("evidence_ids"))
    relationship = str(action.get("relationship", "primary"))
    strategy = str(action.get("strategy", "direct"))
    if (
        not product_ids
        or len(set(product_ids)) != len(product_ids)
        or not set(product_ids).issubset(visible - purchased)
        or not set(evidence_ids).issubset(evidence)
        or not any(item.startswith("product.") for item in evidence_ids)
        or not any(item.startswith(_CONTEXT_EVIDENCE_PREFIXES) for item in evidence_ids)
        or (len(product_ids) == 2 and relationship != "bundle")
        or (relationship == "bundle" and len(product_ids) != 2)
        or (strategy == "bundle" and len(product_ids) != 2)
        or (
            str(action.get("surface", "product_card")) == "bundle"
            and len(product_ids) != 2
        )
        or any(
            not any(
                _evidence_mentions_product(identifier, product_id)
                for identifier in evidence_ids
            )
            for product_id in product_ids
        )
    ):
        return None
    return {
        "decision": "recommend",
        "surface": str(action.get("surface", "product_card")),
        "strategy": strategy,
        "relationship": relationship,
        "product_ids": product_ids,
        "evidence_ids": evidence_ids,
        "explanation": str(action.get("explanation", action.get("reason", "")))[:500],
    }


class TeacherPolicy(RulePolicy):
    """Tool-grounded expert baseline or adapter for an external teacher.

    An optional teacher receives a narrow object exposing only ``observation`` and
    ``call_tool`` and must return an action mapping.  Invalid or ungrounded proposals
    fall back to the deterministic expert heuristic.
    """

    name = "teacher"

    def __init__(
        self,
        teacher: Callable[[_TeacherContext], Any] | Any | None = None,
        *,
        max_recent_interventions: int = 4,
    ) -> None:
        super().__init__(max_recent_interventions=max_recent_interventions)
        self.teacher = teacher

    def act(self, env: Any) -> Action:
        if self.teacher is not None:
            context = _TeacherContext(env.observation, env.call_tool)
            if hasattr(self.teacher, "act"):
                proposal = self.teacher.act(context)
            elif callable(self.teacher):
                proposal = self.teacher(context)
            else:
                raise TypeError("teacher must be callable or expose act(context)")
            normalized = _normalize_teacher_action(proposal, env.observation())
            if normalized is not None:
                return normalized

        # The built-in expert deliberately covers the action families needed by
        # curriculum SFT while remaining a public-state policy.  Its cadence is
        # deterministic and every choice is still grounded in retrieved facts.
        observation = env.observation()
        recent = _recent_interventions(observation)
        step = int(observation.get("step", 0))
        if recent >= self.max_recent_interventions:
            if step % 2:
                return _delay("Recent public history shows frequent interventions.")
            return _no_recommend(
                "Commercial exposure is suppressed after repeated recent interventions."
            )

        rows = _retrieve(env, observation)
        if not rows:
            return _no_recommend("No eligible teacher candidate was retrieved.")
        selected = max(
            rows,
            key=lambda row: (_public_fit_score(row, observation), str(row["product_id"])),
        )
        product_id = selected["product_id"]
        mode = step % 10

        if mode == 0 and _remaining_calls(env.observation()) > 0:
            selected = rows[0]
            product_id = selected["product_id"]
            try:
                result = env.call_tool("find_complements", {"product_id": product_id})
            except (KeyError, RuntimeError, TypeError, ValueError):
                result = {}
            complements = _eligible_rows(_candidate_rows(result), env.observation())
            if complements:
                complement = max(
                    complements,
                    key=lambda row: (
                        _public_fit_score(row, env.observation()),
                        str(row["product_id"]),
                    ),
                )
                return _grounded_recommendation(
                    env,
                    [product_id, complement["product_id"]],
                    surface="bundle",
                    strategy="bundle",
                    relationship="bundle",
                    explanation=(
                        "The retrieved products form a public-evidence complementary bundle."
                    ),
                )

        if mode == 1 and _remaining_calls(env.observation()) > 0:
            selected = rows[0]
            product_id = selected["product_id"]
            try:
                result = env.call_tool("find_alternatives", {"product_id": product_id})
            except (KeyError, RuntimeError, TypeError, ValueError):
                result = {}
            alternatives = [
                row
                for row in _eligible_rows(_candidate_rows(result), env.observation())
                if _number(row.get("price"), float("inf"))
                < _number(selected.get("price"), 0.0)
            ]
            if alternatives:
                alternative = min(
                    alternatives,
                    key=lambda row: (_number(row.get("price")), str(row["product_id"])),
                )
                return _grounded_recommendation(
                    env,
                    [alternative["product_id"]],
                    surface="price_comparison",
                    strategy="cheaper_alternative",
                    relationship="alternative",
                    explanation=(
                        "A lower-priced public alternative preserves the current content fit."
                    ),
                )

        if mode == 2 and _remaining_calls(env.observation()) > 0:
            reviews = _read_reviews(env, product_id)
            return _grounded_recommendation(
                env,
                [product_id],
                surface="review_summary",
                strategy="review_summary",
                explanation=(
                    "The public review summary is shown before conversion"
                    + (" with disclosed risk tags." if reviews.get("risk_tags") else ".")
                ),
            )

        persona = _mapping(observation.get("persona"))
        budget = _number(persona.get("budget"), 0.0)
        if mode == 3 and budget > 0.0:
            selected = max(
                rows,
                key=lambda row: (_number(row.get("price")), str(row["product_id"])),
            )
            product_id = selected["product_id"]
            return _grounded_recommendation(
                env,
                [product_id],
                surface="coupon",
                strategy="discount",
                explanation=(
                    "A coupon reduces the public price relative to the stated initial budget."
                ),
            )

        return _grounded_recommendation(
            env,
            [product_id],
            surface="product_card",
            strategy="direct",
            explanation="Public video, persona, and product facts support this intervention.",
        )


POLICY_REGISTRY = {
    "random": RandomPolicy,
    "popular": PopularPolicy,
    "similarity": SimilarityPolicy,
    "rule": RulePolicy,
    "teacher": TeacherPolicy,
}


def make_policy(name: str, **kwargs: Any) -> FeedPolicy:
    """Construct a registered baseline by its stable lower-case name."""

    try:
        policy_type = POLICY_REGISTRY[str(name).strip().casefold()]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"unknown feed policy {name!r}; expected one of {choices}") from exc
    return policy_type(**kwargs)


def _runtime_reward(raw: Any, fallback_total: Any = 0.0) -> RewardBreakdown:
    values = _mapping(raw)
    if not values:
        return RewardBreakdown(other=_number(fallback_total))
    canonical_names = {
        "qualified_purchase",
        "satisfaction",
        "engagement",
        "revenue",
        "interruption_penalty",
        "return_penalty",
        "irrelevance_penalty",
        "coupon_cost",
        "unsupported_claim_penalty",
        "other",
    }
    if set(values).issubset(canonical_names | {"total"}):
        return RewardBreakdown.from_dict(values)

    qualified = _number(values.get("qualified_purchase_value"))
    satisfaction = _number(values.get("satisfaction")) + _number(
        values.get("purchase_satisfaction")
    )
    engagement = math.fsum(
        _number(values.get(name))
        for name in ("dwell_shaping", "click_shaping", "cart_shaping")
    )
    revenue = _number(values.get("revenue"))
    interruption = _number(values.get("interruption"))
    returned = _number(values.get("return_penalty"))
    irrelevant = _number(values.get("irrelevant_recommendation"))
    coupon = _number(values.get("coupon_cost"))
    unsupported = _number(values.get("unsupported_claim"))
    consumed = {
        "total",
        "qualified_purchase_value",
        "satisfaction",
        "purchase_satisfaction",
        "dwell_shaping",
        "click_shaping",
        "cart_shaping",
        "revenue",
        "interruption",
        "return_penalty",
        "irrelevant_recommendation",
        "coupon_cost",
        "unsupported_claim",
    }
    other = math.fsum(_number(value) for key, value in values.items() if key not in consumed)
    return RewardBreakdown(
        qualified_purchase=qualified,
        satisfaction=satisfaction,
        engagement=engagement,
        revenue=revenue,
        interruption_penalty=interruption,
        return_penalty=returned,
        irrelevance_penalty=irrelevant,
        coupon_cost=coupon,
        unsupported_claim_penalty=unsupported,
        other=other,
    )


_EVENT_ALIASES = {
    "cart": "add_to_cart",
    "cart_withdraw": "remove_from_cart",
}


def _runtime_events(
    raw_events: Any, observation: Mapping[str, Any]
) -> tuple[UserEvent, ...]:
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        return ()
    valid = {event.value for event in EventType}
    video = _mapping(observation.get("current_video"))
    video_id = str(video.get("video_id", "")) or None
    converted: list[UserEvent] = []
    for raw in raw_events:
        row = _mapping(raw)
        event_type = _EVENT_ALIASES.get(_event_type(row), _event_type(row))
        if event_type not in valid:
            continue
        metadata = _mapping(row.get("metadata"))
        for field in ("event_id", "source_step"):
            if field in row:
                metadata[field] = row[field]
        value = _number(row.get("value"))
        converted.append(
            UserEvent(
                event_type=event_type,
                step=max(int(_number(row.get("step"))), 0),
                video_id=str(metadata.get("video_id", video_id or "")) or None,
                product_id=str(row.get("product_id", "")) or None,
                dwell_seconds=value if event_type in {"watch", "skip"} else 0.0,
                value=value,
                count=max(int(_number(row.get("count"), 1.0)), 1),
                metadata=metadata,
            )
        )
    return tuple(converted)


def _runtime_tools(raw_records: Any, step: int) -> tuple[ToolRecord, ...]:
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return ()
    converted: list[ToolRecord] = []
    for raw in raw_records:
        row = _mapping(raw)
        result = row.get("result")
        result_row = _mapping(result)
        converted.append(
            ToolRecord(
                tool_name=str(row.get("tool_name", row.get("name", "unknown"))),
                arguments=_mapping(row.get("arguments", row.get("parameters"))),
                result=result,
                evidence_ids=tuple(_strings(result_row.get("evidence_ids"))),
                step=max(int(_number(row.get("step"), step)), 0),
                call_id=str(row.get("call_id", row.get("id", ""))),
                error=(str(row["error"]) if row.get("error") else None),
            )
        )
    return tuple(converted)


def transition_from_step_result(
    *,
    step: int,
    observation: Mapping[str, Any],
    action: Any,
    step_result: Any,
) -> FeedTransition:
    """Convert one simulator result without discarding Feed-specific diagnostics."""

    action_row = _mapping(action)
    typed_action = FeedAction.from_dict(action_row)
    raw_events = getattr(step_result, "events", None)
    if raw_events is None:
        raw_events = _mapping(step_result).get("events", ())
    raw_reward = getattr(step_result, "reward_breakdown", None)
    if raw_reward is None:
        raw_reward = _mapping(step_result).get(
            "reward_breakdown", _mapping(step_result).get("reward", {})
        )
    info = getattr(step_result, "info", None)
    if info is None:
        info = _mapping(step_result).get("info", {})
    info_row = _mapping(info)
    tool_records = info_row.get("tool_records", ())
    post_observation = getattr(step_result, "observation", None)
    if post_observation is None:
        post_observation = _mapping(step_result).get("observation", {})
    done = getattr(step_result, "done", None)
    if done is None:
        done = bool(_mapping(step_result).get("done", False))
    reward_total = getattr(step_result, "reward", None)
    reward_total = _number(reward_total, _number(_mapping(raw_reward).get("total")))
    video = _mapping(observation.get("current_video"))
    video_id = str(video.get("video_id", f"step-{step}"))
    return FeedTransition(
        step=step,
        video_id=video_id,
        action=typed_action,
        events=_runtime_events(raw_events, observation),
        reward=_runtime_reward(raw_reward, reward_total),
        tool_records=_runtime_tools(tool_records, step),
        observation=dict(observation),
        done=bool(done),
        metadata={
            "post_observation": _mapping(post_observation),
            "raw_events": [_mapping(event) for event in raw_events or ()],
            "raw_reward_breakdown": _mapping(raw_reward),
            "runtime_info": info_row,
        },
    )


def _policy_action(policy: Any, env: Any) -> Any:
    if hasattr(policy, "act"):
        return policy.act(env)
    if hasattr(policy, "select_action"):
        return policy.select_action(env)
    if callable(policy):
        return policy(env)
    raise TypeError("policy must be callable or expose act(env)")


def rollout_episode(
    env: Any,
    policy: Any,
    *,
    max_steps: int | None = None,
) -> EpisodeResult:
    """Run one policy to completion using the environment's public interaction API."""

    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer or None")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

    initial = env.observation()
    episode_id = str(initial.get("episode_id", "episode"))
    transitions: list[FeedTransition] = []
    state = initial
    while not state.get("done") and (max_steps is None or len(transitions) < max_steps):
        step = int(state.get("step", len(transitions)))
        action = _policy_action(policy, env)
        result = env.step(action)
        transition = transition_from_step_result(
            step=step,
            observation=state,
            action=action,
            step_result=result,
        )
        transitions.append(transition)
        state = _mapping(getattr(result, "observation", None)) or _mapping(result).get(
            "observation", {}
        )

    done = bool(state.get("done", transitions[-1].done if transitions else initial.get("done")))
    termination_reason = "feed_exhausted" if done else "max_steps"
    final_state = dict(state)
    if done:
        try:
            summary = env.summary()
        except AttributeError:
            summary = None
        if isinstance(summary, Mapping):
            final_state["evaluation_summary"] = dict(summary)
    policy_name = str(getattr(policy, "name", type(policy).__name__))
    return EpisodeResult(
        episode_id=episode_id,
        transitions=tuple(transitions),
        done=done,
        termination_reason=termination_reason,
        final_state=final_state,
        metadata={"policy": policy_name},
    )


__all__ = [
    "FeedPolicy",
    "POLICY_REGISTRY",
    "PopularPolicy",
    "RandomPolicy",
    "RulePolicy",
    "SimilarityPolicy",
    "TeacherPolicy",
    "make_policy",
    "rollout_episode",
    "transition_from_step_result",
]
