"""Deterministic long-horizon Feed POMDP for shopping interventions.

The environment keeps user intent, trust, fatigue and transition probabilities
private.  Policies only receive the allow-listed public observation and evidence
returned by information tools.  A recommendation is a non-terminal intervention:
delayed purchases and returns are settled later in the same fixed feed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import islice
import math
from typing import Any, Mapping

from shopping_grpo.feed.evidence import evidence_component, has_product_evidence
from shopping_grpo.feed.randomness import CommonRandomNumbers, sigmoid


FEED_ENVIRONMENT_VERSION = "feed-environment-v1"
FEED_REWARD_VERSION = "feed-reward-v1"
INFORMATION_TOOLS = {
    "retrieve_products",
    "inspect_product",
    "compare_products",
    "read_reviews",
    "find_alternatives",
    "find_complements",
    "check_inventory",
}
DEFAULT_RETRIEVAL_LIMIT = 5
DEFAULT_RELATIONSHIP_LIMIT = 4
MAX_IDENTIFIER_CHARS = 128


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item))


def _clipped_text(value: Any, maximum: int) -> str:
    return str(value or "")[: int(maximum)]


def _clipped_strings(
    value: Any,
    *,
    maximum_items: int,
    maximum_chars: int,
) -> list[str]:
    return [
        _clipped_text(item, maximum_chars)
        for item in _strings(value)[: int(maximum_items)]
    ]


def _identifier(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > MAX_IDENTIFIER_CHARS:
        raise ValueError(
            f"{name} exceeds {MAX_IDENTIFIER_CHARS} characters"
        )
    return text


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(float(value), lower), upper)


@dataclass
class LatentUserState:
    """Private numerical state.  This object must never enter an observation."""

    short_term_interest: dict[str, float]
    purchase_intent: float
    trust: float
    fatigue: float
    budget_remaining: float
    price_sensitivity: float
    satisfaction: float = 0.5


@dataclass
class PendingPurchase:
    product_id: str
    source_step: int
    due_step: int
    relevance: float
    hard_match: bool
    soft_satisfaction: float
    risk_disclosed: bool
    coupon_cost: float


@dataclass
class PurchaseRecord:
    product_id: str
    source_step: int
    price: float
    hard_match: bool
    soft_satisfaction: float
    risk_disclosed: bool
    returned: bool = False


@dataclass
class FeedStepResult:
    observation: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, float]
    events: list[dict[str, Any]]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "reward_breakdown": self.reward_breakdown,
            "events": self.events,
            "done": self.done,
            "info": self.info,
        }


class FeedActionGuardError(ValueError):
    """A commit action violates the current public-state contract."""


class FeedShoppingEnv:
    """In-process Feed POMDP with a fixed video sequence and hybrid user model."""

    def __init__(
        self,
        episode_seed: Any,
        catalog: Any,
        *,
        calibration: Any = None,
        max_info_calls_per_video: int = 3,
        recent_event_limit: int = 24,
    ) -> None:
        if max_info_calls_per_video < 0:
            raise ValueError("max_info_calls_per_video must be non-negative")
        self.seed = episode_seed
        self.catalog = catalog
        self.calibration = calibration
        self.max_info_calls_per_video = int(max_info_calls_per_video)
        self.recent_event_limit = int(recent_event_limit)
        self.episode_id = _identifier(
            _get(episode_seed, "episode_id", "episode"), name="episode_id"
        )
        random_seed = _get(episode_seed, "seed", _get(episode_seed, "random_seed", 0))
        self.noise = CommonRandomNumbers(random_seed, self.episode_id)
        self.persona = _get(episode_seed, "persona", {})
        self.persona_id = _identifier(
            _get(self.persona, "persona_id", _get(self.persona, "user_id", "user")),
            name="persona_id",
        )
        self.videos = list(_get(episode_seed, "videos", _get(episode_seed, "feed", ())))
        self.product_ids = tuple(
            _identifier(item, name="product_id")
            for item in _get(episode_seed, "product_ids", ())
        )
        for video in self.videos:
            _identifier(_get(video, "video_id", ""), name="video_id")
        raw_inventory = _get(episode_seed, "inventory", {}) or {}
        inventory_pairs = (
            raw_inventory.items() if isinstance(raw_inventory, Mapping) else raw_inventory
        )
        self._initial_inventory = {
            _identifier(key, name="inventory product_id"): int(value)
            for key, value in inventory_pairs
        }
        self._initial_promotions = dict(_get(episode_seed, "promotions", {}) or {})
        if not self.videos:
            raise ValueError("episode_seed must contain at least one video")
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Restore the episode and return the first public observation."""
        category_interest = _get(
            self.persona,
            "category_interests",
            _get(self.persona, "category_interest", {}),
        )
        if isinstance(category_interest, Mapping):
            interest = {str(key): _bounded(float(value)) for key, value in category_interest.items()}
        else:
            interest = {str(key): 0.65 for key in _strings(category_interest)}
        raw_budget = _get(self.persona, "budget", _get(self.persona, "budget_limit", 300.0))
        budget = 300.0 if raw_budget is None else float(raw_budget)
        self._latent = LatentUserState(
            short_term_interest=interest,
            purchase_intent=_bounded(float(_get(self.persona, "initial_intent", 0.18))),
            trust=_bounded(float(_get(self.persona, "initial_trust", 0.62))),
            fatigue=0.0,
            budget_remaining=max(budget, 0.0),
            price_sensitivity=_bounded(float(_get(self.persona, "price_sensitivity", 0.5))),
        )
        self.step_index = 0
        self.done = False
        self.inventory = dict(self._initial_inventory)
        for product_id in self.product_ids:
            self.inventory.setdefault(product_id, 10)
        self.promotions = dict(self._initial_promotions)
        self._session_breaks = {
            int(step) for step in (_get(self.seed, "session_breaks", ()) or ())
        }
        self.cart: list[str] = []
        self.purchases: list[PurchaseRecord] = []
        self.pending: list[PendingPurchase] = []
        self.history: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.total_reward = 0.0
        self.net_revenue = 0.0
        self.info_tool_calls = 0
        self.tool_records: list[dict[str, Any]] = []
        self._step_tool_records: list[dict[str, Any]] = []
        self._visible_product_ids: set[str] = set()
        self._evidence: dict[str, dict[str, Any]] = {}
        self._exposure_counts: dict[str, int] = {}
        self._event_counter = 0
        self._build_base_evidence()
        return self.observation()

    @property
    def current_video(self) -> Any:
        if self.done or self.step_index >= len(self.videos):
            return None
        return self.videos[self.step_index]

    def observation(self) -> dict[str, Any]:
        """Return an allow-listed public state with no latent values or probabilities."""
        video = self.current_video
        raw_budget = _get(self.persona, "budget", _get(self.persona, "budget_limit", 300.0))
        public_persona = {
            "persona_id": self.persona_id,
            "budget": 300.0 if raw_budget is None else float(raw_budget),
            "category_interests": self._public_category_interests(),
            "style_preferences": _clipped_strings(
                _get(self.persona, "style_preferences", _get(self.persona, "styles", ())),
                maximum_items=32,
                maximum_chars=160,
            ),
        }
        return {
            "observation_version": "feed-observation-v1",
            "environment_version": FEED_ENVIRONMENT_VERSION,
            "episode_id": self.episode_id,
            "step": self.step_index,
            "total_steps": len(self.videos),
            "persona": public_persona,
            # Keep simulator labels (related products, embeddings and seed metadata)
            # evaluator-only.  Exposing those fields would turn retrieval into a
            # supervised lookup instead of a partially observed decision problem.
            "current_video": self._public_video(video) if video is not None else None,
            "recent_events": [
                self._public_event(event)
                for event in self.history[-self.recent_event_limit :]
            ],
            "cart": list(self.cart),
            "purchased_product_ids": [record.product_id for record in self.purchases],
            "visible_product_ids": sorted(self._visible_product_ids),
            "evidence_ids": sorted(self._evidence),
            "info_tool_calls": self.info_tool_calls,
            "max_info_tool_calls": self.max_info_calls_per_video,
            "done": self.done,
        }

    def render_observation(self) -> str:
        """Render through the canonical allow-list when it is available."""
        try:
            from shopping_grpo.feed.observation import render_feed_observation
        except ImportError:
            import json

            return json.dumps(self.observation(), ensure_ascii=False, sort_keys=True)
        return render_feed_observation(self.observation())

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Execute an information tool or the final ``commit_recommendation`` tool."""
        name = str(name)
        arguments = dict(arguments or {})
        if self.done:
            raise RuntimeError("feed episode is already terminal")
        if name == "commit_recommendation":
            result = self.step(arguments)
            return {
                "observation": result.observation,
                "events": [self._public_event(event) for event in result.events],
                "done": result.done,
            }
        if name not in INFORMATION_TOOLS:
            raise ValueError(f"unknown Feed tool: {name}")
        if self.info_tool_calls >= self.max_info_calls_per_video:
            raise FeedActionGuardError("max_info_tool_calls_exceeded")

        dispatcher = {
            "retrieve_products": self._retrieve_products,
            "inspect_product": self._inspect_product,
            "compare_products": self._compare_products,
            "read_reviews": self._read_reviews,
            "find_alternatives": self._find_alternatives,
            "find_complements": self._find_complements,
            "check_inventory": self._check_inventory,
        }
        result = dispatcher[name](arguments)
        self.info_tool_calls += 1
        record = {
            "step": self.step_index,
            "tool_name": name,
            "arguments": arguments,
            "result": result,
        }
        self.tool_records.append(record)
        self._step_tool_records.append(record)
        return result

    def step(self, raw_action: Any) -> FeedStepResult:
        """Commit exactly one When-What-How intervention and advance the fixed feed."""
        if self.done:
            raise RuntimeError("feed episode is already terminal")
        action = self._action_dict(raw_action)
        self._guard_action(action)
        source_step = self.step_index
        events: list[dict[str, Any]] = []
        reward = self._empty_reward()

        if source_step in self._session_breaks:
            # A break reduces accumulated interruption pressure but preserves
            # learned interests and purchases across sessions.
            self._latent.fatigue = _bounded(self._latent.fatigue * 0.35)
            self._latent.purchase_intent = _bounded(self._latent.purchase_intent * 0.9)
            events.append(self._event("session_break", source_step=source_step))
        self._simulate_content_consumption(events, reward)
        decision = action["decision"]
        if decision == "recommend":
            self._simulate_recommendation(action, events, reward)
        elif decision == "delay":
            self._latent.fatigue = _bounded(self._latent.fatigue - 0.05)
            self._latent.trust = _bounded(self._latent.trust + 0.01)
            reward["satisfaction"] += 0.03
            events.append(self._event("delay", source_step=source_step))
        else:
            readiness = self._readiness(self._video_relevance())
            correct = readiness < 0.48 or self._latent.fatigue > 0.58
            reward["correct_no_recommend"] += 0.08 if correct else -0.02
            self._latent.fatigue = _bounded(self._latent.fatigue - 0.06)
            self._latent.satisfaction = _bounded(
                self._latent.satisfaction + (0.025 if correct else -0.005)
            )
            events.append(
                self._event(
                    "no_recommend",
                    source_step=source_step,
                    metadata={"hindsight_correct": correct},
                )
            )

        self._settle_pending(source_step, events, reward, force=False)
        tool_records = list(self._step_tool_records)
        self._step_tool_records = []
        self.step_index += 1
        self.info_tool_calls = 0
        self._visible_product_ids.clear()
        self._evidence.clear()
        terminal = self.step_index >= len(self.videos)
        if terminal:
            self._settle_pending(self.step_index, events, reward, force=True)
            self._settle_returns(events, reward)
            self.done = True
        self.history.extend(events)
        if not terminal:
            self._build_base_evidence()

        reward["total"] = sum(value for key, value in reward.items() if key != "total")
        self.total_reward += reward["total"]
        transition = {
            "step": source_step,
            "observation": self.observation() if not terminal else self._terminal_observation(),
            "action": action,
            "tools": tool_records,
            "events": events,
            "reward": dict(reward),
            "done": terminal,
        }
        self.transitions.append(transition)
        return FeedStepResult(
            observation=transition["observation"],
            reward=reward["total"],
            reward_breakdown=reward,
            events=events,
            done=terminal,
            info={
                "reward_version": FEED_REWARD_VERSION,
                "source_step": source_step,
                "tool_records": tool_records,
                "episode_return": self.total_reward,
            },
        )

    def summary(self) -> dict[str, Any]:
        """Return evaluator-only terminal diagnostics; this is never an observation."""
        counts: dict[str, int] = {}
        for transition in self.transitions:
            for event in transition["events"]:
                event_type = str(event["event_type"])
                counts[event_type] = counts.get(event_type, 0) + 1
        return {
            "episode_id": self.episode_id,
            "environment_version": FEED_ENVIRONMENT_VERSION,
            "reward_version": FEED_REWARD_VERSION,
            "steps": len(self.transitions),
            "done": self.done,
            "episode_return": self.total_reward,
            "net_revenue": self.net_revenue,
            "event_counts": counts,
            "terminal_fatigue": self._latent.fatigue,
            "terminal_satisfaction": self._latent.satisfaction,
            "purchases": [asdict(record) for record in self.purchases],
        }

    def _terminal_observation(self) -> dict[str, Any]:
        state = self.observation()
        state["current_video"] = None
        state["done"] = True
        return state

    def _action_dict(self, raw_action: Any) -> dict[str, Any]:
        action = _plain(raw_action)
        if not isinstance(action, Mapping):
            raise FeedActionGuardError("action_must_be_object")
        allowed = {
            "decision",
            "surface",
            "strategy",
            "relationship",
            "product_ids",
            "evidence_ids",
            "explanation",
            "reason",
        }
        extra = sorted(set(action) - allowed)
        if extra:
            raise FeedActionGuardError("schema_extra_arguments:" + ",".join(extra))
        return {
            "decision": _enum_value(action.get("decision", "")),
            "surface": _enum_value(action.get("surface", "none")),
            "strategy": _enum_value(action.get("strategy", "none")),
            "relationship": _enum_value(action.get("relationship", "primary")),
            "product_ids": [str(item) for item in action.get("product_ids", ())],
            "evidence_ids": [str(item) for item in action.get("evidence_ids", ())],
            "explanation": str(action.get("explanation", action.get("reason", ""))),
        }

    def _guard_action(self, action: dict[str, Any]) -> None:
        decision = action["decision"]
        if decision not in {"recommend", "delay", "no_recommend"}:
            raise FeedActionGuardError("invalid_decision")
        if decision != "recommend":
            if action["product_ids"]:
                raise FeedActionGuardError("non_recommend_has_products")
            if action["surface"] not in {"none", ""}:
                raise FeedActionGuardError("non_recommend_has_surface")
            if action["strategy"] != "none":
                raise FeedActionGuardError("non_recommend_has_strategy")
            if action["relationship"] != "primary":
                raise FeedActionGuardError("non_recommend_has_relationship")
            return
        if not 1 <= len(action["product_ids"]) <= 2:
            raise FeedActionGuardError("recommend_requires_one_or_two_products")
        if len(set(action["product_ids"])) != len(action["product_ids"]):
            raise FeedActionGuardError("duplicate_product_ids")
        if not set(action["product_ids"]).issubset(self._visible_product_ids):
            raise FeedActionGuardError("product_not_visible")
        if not action["evidence_ids"]:
            raise FeedActionGuardError("missing_evidence")
        if action["strategy"] == "none":
            raise FeedActionGuardError("recommend_requires_strategy")
        if not set(action["evidence_ids"]).issubset(self._evidence):
            raise FeedActionGuardError("unknown_evidence")
        relationship = action["relationship"]
        if len(action["product_ids"]) == 2 and relationship != "bundle":
            raise FeedActionGuardError("multi_product_requires_bundle_relationship")
        if relationship == "bundle" and len(action["product_ids"]) != 2:
            raise FeedActionGuardError("bundle_relationship_requires_two_products")
        if action["strategy"] == "bundle" and len(action["product_ids"]) != 2:
            raise FeedActionGuardError("bundle_strategy_requires_two_products")
        if action["surface"] == "bundle" and len(action["product_ids"]) != 2:
            raise FeedActionGuardError("bundle_surface_requires_two_products")
        for product_id in action["product_ids"]:
            if not has_product_evidence(action["evidence_ids"], product_id):
                raise FeedActionGuardError(f"missing_product_evidence:{product_id}")
        if not any(
            evidence_id.startswith(("video.", "history.", "persona."))
            for evidence_id in action["evidence_ids"]
        ):
            raise FeedActionGuardError("missing_context_evidence")
        if action["surface"] not in {
            "product_card",
            "coupon",
            "review_summary",
            "price_comparison",
            "similar_products",
            "bundle",
            "creator_video",
        }:
            raise FeedActionGuardError("invalid_recommendation_surface")
        purchased = {record.product_id for record in self.purchases}
        if purchased.intersection(action["product_ids"]):
            raise FeedActionGuardError("repeat_marketing_after_purchase")

    def _retrieve_products(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("retrieve_products requires a non-empty query")
        video = self.current_video
        if hasattr(self.catalog, "search"):
            try:
                products = self.catalog.search(
                    query,
                    video=video,
                    persona=self.persona,
                    limit=DEFAULT_RETRIEVAL_LIMIT,
                    candidate_ids=self.product_ids or None,
                )
            except TypeError:
                products = self.catalog.search(
                    query,
                    limit=DEFAULT_RETRIEVAL_LIMIT,
                )
        else:
            products = []
        product_rows = []
        evidence = []
        for product in islice(iter(products or ()), DEFAULT_RETRIEVAL_LIMIT):
            row = self._product_public(product, detailed=False)
            product_id = row["product_id"]
            if self.product_ids and product_id not in self.product_ids:
                continue
            self._visible_product_ids.add(product_id)
            evidence_id = f"product.{evidence_component(product_id)}.retrieved"
            self._register_evidence(evidence_id, row)
            evidence.append(evidence_id)
            product_rows.append(row)
        return {"products": product_rows, "evidence_ids": evidence}

    def _inspect_product(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_id = self._require_visible_product(arguments.get("product_id"))
        product = self._catalog_get(product_id)
        row = self._product_public(product, detailed=True)
        evidence = []
        for field_name in ("title", "category", "price", "attributes", "rating"):
            evidence_id = f"product.{evidence_component(product_id)}.{field_name}"
            self._register_evidence(evidence_id, row.get(field_name))
            evidence.append(evidence_id)
        return {"product": row, "evidence_ids": evidence}

    def _compare_products(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_ids = [self._require_visible_product(item) for item in arguments.get("product_ids", ())]
        if len(product_ids) < 2:
            raise ValueError("compare_products requires at least two visible product IDs")
        rows = [self._product_public(self._catalog_get(product_id), detailed=True) for product_id in product_ids]
        evidence_id = "product.compare." + ".".join(
            evidence_component(product_id) for product_id in product_ids
        )
        comparison = [
            {
                "product_id": row["product_id"],
                "price": row["price"],
                "category": row["category"],
                "rating": row["rating"],
            }
            for row in rows
        ]
        self._register_evidence(evidence_id, comparison)
        return {"comparison": comparison, "evidence_ids": [evidence_id]}

    def _read_reviews(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_id = self._require_visible_product(arguments.get("product_id"))
        product = self._catalog_get(product_id)
        summary = _clipped_text(
            _get(product, "review_summary", "No structured review summary available."),
            1200,
        )
        risk_tags = _clipped_strings(
            _get(product, "risk_tags", _get(product, "tags", ())),
            maximum_items=16,
            maximum_chars=160,
        )
        evidence_id = f"product.{evidence_component(product_id)}.reviews"
        self._register_evidence(evidence_id, {"summary": summary, "risk_tags": risk_tags})
        return {"product_id": product_id, "summary": summary, "risk_tags": risk_tags, "evidence_ids": [evidence_id]}

    def _find_alternatives(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_id = self._require_visible_product(arguments.get("product_id"))
        if hasattr(self.catalog, "alternatives"):
            products = self.catalog.alternatives(
                product_id,
                limit=DEFAULT_RELATIONSHIP_LIMIT,
            )
        else:
            products = []
        return self._relationship_result("alternatives", product_id, products)

    def _find_complements(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_id = self._require_visible_product(arguments.get("product_id"))
        if hasattr(self.catalog, "complements"):
            products = self.catalog.complements(
                product_id,
                limit=DEFAULT_RELATIONSHIP_LIMIT,
            )
        else:
            products = []
        return self._relationship_result("complements", product_id, products)

    def _check_inventory(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        product_ids = [self._require_visible_product(item) for item in arguments.get("product_ids", ())]
        if not product_ids:
            raise ValueError("check_inventory requires product_ids")
        inventory = {product_id: int(self.inventory.get(product_id, 0)) for product_id in product_ids}
        evidence = []
        for product_id, quantity in inventory.items():
            evidence_id = f"product.{evidence_component(product_id)}.inventory"
            self._register_evidence(evidence_id, quantity)
            evidence.append(evidence_id)
        return {"inventory": inventory, "evidence_ids": evidence}

    def _relationship_result(self, relation: str, source_id: str, products: Any) -> dict[str, Any]:
        rows = []
        evidence = []
        for product in islice(iter(products or ()), DEFAULT_RELATIONSHIP_LIMIT):
            row = self._product_public(product, detailed=False)
            product_id = row["product_id"]
            if self.product_ids and product_id not in self.product_ids:
                continue
            self._visible_product_ids.add(product_id)
            evidence_id = (
                f"product.{evidence_component(source_id)}.{relation}."
                f"{evidence_component(product_id)}"
            )
            self._register_evidence(evidence_id, row)
            evidence.append(evidence_id)
            rows.append(row)
        return {relation: rows, "evidence_ids": evidence}

    def _simulate_content_consumption(self, events: list[dict[str, Any]], reward: dict[str, float]) -> None:
        video = self.current_video
        relevance = self._video_relevance()
        raw_duration = _get(video, "duration_seconds", _get(video, "duration", 30.0))
        duration = max(30.0 if raw_duration is None else float(raw_duration), 1.0)
        mean_dwell = self._calibration_value("mean_dwell_seconds", 8.0)
        dwell_shift = mean_dwell / duration - 8.0 / 30.0
        dwell_fraction = _bounded(
            0.12 + 0.72 * relevance - 0.23 * self._latent.fatigue
            + dwell_shift
            + 0.08 * self.noise.normal(self.step_index, "dwell")
        )
        dwell_seconds = round(duration * dwell_fraction, 3)
        watch_prior = self._calibration_value("watch_probability", 0.55)
        skipped = dwell_fraction < _bounded(0.24 + (0.55 - watch_prior) * 0.35, 0.08, 0.5)
        liked = self.noise.bernoulli(
            sigmoid(
                self._coef("like_bias", -2.3)
                + 4.0 * relevance
                - 1.2 * self._latent.fatigue
            ),
            self.step_index,
            "like",
        )
        event_type = "skip" if skipped else "watch"
        events.append(self._event(event_type, value=dwell_seconds, metadata={"video_id": str(_get(video, "video_id", self.step_index))}))
        if liked:
            events.append(self._event("like", metadata={"video_id": str(_get(video, "video_id", self.step_index))}))
        reward["dwell_shaping"] += min(dwell_fraction, 1.0) * 0.02
        categories = self._video_categories()
        increment = (0.08 if liked else 0.035) * (1.0 if not skipped else 0.35)
        for category in categories:
            current = self._latent.short_term_interest.get(category, 0.25)
            self._latent.short_term_interest[category] = _bounded(current + increment)
        self._latent.purchase_intent = _bounded(
            self._latent.purchase_intent + 0.09 * relevance * dwell_fraction + (0.04 if liked else 0.0) - 0.025 * self._latent.fatigue
        )

    def _simulate_recommendation(self, action: dict[str, Any], events: list[dict[str, Any]], reward: dict[str, float]) -> None:
        product_ids = action["product_ids"]
        primary = self._catalog_get(product_ids[0])
        source_step = self.step_index
        relevance = self._product_relevance(primary)
        hard_match = self._hard_match(primary)
        soft_satisfaction = self._soft_satisfaction(primary)
        price_fit = self._price_fit(primary)
        strategy_fit = self._strategy_fit(action, primary)
        repeated = self._exposure_counts.get(product_ids[0], 0) > 0
        self._exposure_counts[product_ids[0]] = self._exposure_counts.get(product_ids[0], 0) + 1
        risk_disclosed = action["surface"] == "review_summary" or action["strategy"] == "review_summary"

        events.append(self._event("intervention", product_id=product_ids[0], source_step=source_step))
        if repeated:
            reward["repeat_exposure"] -= 0.18
            events.append(self._event("repeat_exposure", product_id=product_ids[0], source_step=source_step))
        if relevance < 0.34 or not hard_match:
            reward["irrelevant_recommendation"] -= 0.32
        readiness = self._readiness(relevance)
        if readiness < 0.43 or self._latent.fatigue > 0.62:
            reward["interruption"] -= 0.16 + 0.12 * self._latent.fatigue
        coupon_cost = 0.0
        if action["surface"] == "coupon":
            coupon_cost = min(0.2, 0.03 + 0.07 * self._latent.price_sensitivity)
            reward["coupon_cost"] -= coupon_cost

        click_probability = sigmoid(
            self._coef("click_bias", -2.1)
            + self._coef("click_relevance", 2.5) * relevance
            + self._coef("click_intent", 1.5) * self._latent.purchase_intent
            + self._coef("click_price_fit", 0.8) * price_fit
            + self._coef("click_strategy_fit", 0.55) * strategy_fit
            + 0.45 * self._latent.trust
            - self._coef("click_fatigue", 1.65) * self._latent.fatigue
        )
        clicked = self.noise.bernoulli(click_probability, source_step, "click", product_ids[0])
        if clicked:
            events.append(self._event("click", product_id=product_ids[0], source_step=source_step))
            reward["click_shaping"] += 0.04
            cart_probability = sigmoid(
                self._coef("cart_bias", -1.7)
                + 1.9 * relevance
                + 1.5 * self._latent.purchase_intent
                + 0.9 * price_fit
                + 0.35 * self._latent.trust
                - 0.7 * self._latent.fatigue
            )
            carted = self.noise.bernoulli(cart_probability, source_step, "cart", product_ids[0])
            if carted and self.inventory.get(product_ids[0], 0) > 0:
                if product_ids[0] not in self.cart:
                    self.cart.append(product_ids[0])
                events.append(self._event("cart", product_id=product_ids[0], source_step=source_step))
                reward["cart_shaping"] += 0.08
                buy_probability = sigmoid(
                    self._coef("buy_bias", -1.8)
                    + 1.7 * soft_satisfaction
                    + 1.3 * self._latent.purchase_intent
                    + 0.8 * price_fit
                    + 0.55 * self._latent.trust
                    - 0.65 * self._latent.fatigue
                )
                if self.noise.bernoulli(buy_probability, source_step, "buy", product_ids[0]):
                    delay = self.noise.integer(1, 3, source_step, "purchase_delay", product_ids[0])
                    self.pending.append(
                        PendingPurchase(
                            product_id=product_ids[0],
                            source_step=source_step,
                            due_step=source_step + delay,
                            relevance=relevance,
                            hard_match=hard_match,
                            soft_satisfaction=soft_satisfaction,
                            risk_disclosed=risk_disclosed,
                            coupon_cost=coupon_cost,
                        )
                    )
        self._latent.fatigue = _bounded(self._latent.fatigue + (0.17 if not clicked else 0.10))
        self._latent.trust = _bounded(
            self._latent.trust
            + (0.035 if relevance >= 0.6 and hard_match else -0.055)
            + (0.02 if risk_disclosed else 0.0)
        )

        if len(product_ids) == 2:
            complement_ok = self._are_complements(product_ids[0], product_ids[1])
            if complement_ok:
                reward["bundle_value"] += 0.06
                events.append(self._event("bundle_offer", product_id=product_ids[1], source_step=source_step, metadata={"complementary": True}))
            else:
                reward["irrelevant_recommendation"] -= 0.22

    def _settle_pending(self, step: int, events: list[dict[str, Any]], reward: dict[str, float], *, force: bool) -> None:
        remaining = []
        for pending in self.pending:
            if not force and pending.due_step > step:
                remaining.append(pending)
                continue
            product = self._catalog_get(pending.product_id)
            price = float(_get(product, "price", 0.0))
            if self.inventory.get(pending.product_id, 0) <= 0 or price > self._latent.budget_remaining:
                events.append(self._event("cart_withdraw", product_id=pending.product_id, source_step=pending.source_step))
                continue
            self.inventory[pending.product_id] -= 1
            self._latent.budget_remaining = max(0.0, self._latent.budget_remaining - price)
            if pending.product_id in self.cart:
                self.cart.remove(pending.product_id)
            record = PurchaseRecord(
                product_id=pending.product_id,
                source_step=pending.source_step,
                price=price,
                hard_match=pending.hard_match,
                soft_satisfaction=pending.soft_satisfaction,
                risk_disclosed=pending.risk_disclosed,
            )
            self.purchases.append(record)
            qualified = float(pending.hard_match) * pending.soft_satisfaction
            value = min(price / 100.0, 4.0) * qualified
            satisfaction_credit = 0.12 * pending.soft_satisfaction
            reward["qualified_purchase_value"] += value
            reward["purchase_satisfaction"] += satisfaction_credit
            self.net_revenue += price - pending.coupon_cost * price
            events.append(
                self._event(
                    "purchase",
                    product_id=pending.product_id,
                    source_step=pending.source_step,
                    value=price,
                    metadata={
                        "qualified": bool(pending.hard_match and pending.soft_satisfaction >= 0.5),
                        "realized_at_step": step,
                        # Evaluator-only component amounts let delayed credit move
                        # all purchase value without stealing same-step satisfaction.
                        "qualified_purchase_credit": value,
                        "satisfaction_credit": satisfaction_credit,
                    },
                )
            )
            self._latent.purchase_intent = _bounded(self._latent.purchase_intent - 0.45)
            self._latent.satisfaction = _bounded(self._latent.satisfaction + 0.15 * pending.soft_satisfaction)
        self.pending = remaining

    def _settle_returns(self, events: list[dict[str, Any]], reward: dict[str, float]) -> None:
        for record in self.purchases:
            product = self._catalog_get(record.product_id)
            risk = min(
                0.45,
                0.06
                + 0.07
                * len(_strings(_get(product, "risk_tags", _get(product, "tags", ())))),
            )
            risk += self._calibration_value("return_probability", 0.08) - 0.08
            return_probability = _bounded(
                risk
                + 0.38 * (1.0 - record.soft_satisfaction)
                + (0.28 if not record.hard_match else 0.0)
                - (0.12 if record.risk_disclosed else 0.0)
                - 0.08 * self._latent.trust
            )
            returned = self.noise.bernoulli(
                return_probability,
                len(self.videos),
                "return",
                f"{record.product_id}:{record.source_step}",
            )
            if not returned:
                events.append(self._event("retained", product_id=record.product_id, source_step=record.source_step))
                continue
            record.returned = True
            penalty = min(record.price / 80.0, 5.0)
            reward["return_penalty"] -= penalty
            self.net_revenue -= record.price
            self._latent.satisfaction = _bounded(self._latent.satisfaction - 0.22)
            events.append(
                self._event(
                    "return",
                    product_id=record.product_id,
                    source_step=record.source_step,
                    value=record.price,
                    metadata={"realized_at_step": len(self.videos)},
                )
            )

    def _video_relevance(self) -> float:
        categories = self._video_categories()
        category_score = max(
            (self._latent.short_term_interest.get(category, 0.2) for category in categories),
            default=0.2,
        )
        preferred_styles = set(
            _strings(_get(self.persona, "style_preferences", _get(self.persona, "styles", ())))
        )
        video_styles = set(_strings(_get(self.current_video, "style", _get(self.current_video, "styles", ()))))
        style_score = len(preferred_styles & video_styles) / max(len(video_styles), 1)
        return _bounded(0.78 * category_score + 0.22 * style_score)

    def _video_categories(self) -> set[str]:
        video = self.current_video
        categories = set(_strings(_get(video, "categories", ())))
        for product_id in _strings(_get(video, "related_products", _get(video, "related_product_ids", ()))):
            try:
                categories.add(str(_get(self._catalog_get(product_id), "category", "")))
            except (KeyError, ValueError):
                pass
        categories.discard("")
        return categories

    def _product_relevance(self, product: Any) -> float:
        product_id = _identifier(
            _get(product, "product_id", _get(product, "asin", "")),
            name="product_id",
        )
        related = set(_strings(_get(self.current_video, "related_products", _get(self.current_video, "related_product_ids", ()))))
        direct = 1.0 if product_id in related else 0.0
        video_terms = set(_strings(_get(self.current_video, "objects", ()))) | set(self._video_categories())
        product_terms = set(_strings(_get(product, "attributes", _get(product, "attribute", ()))))
        product_terms.add(str(_get(product, "category", "")))
        overlap = len(video_terms & product_terms) / max(len(video_terms), 1)
        return _bounded(0.72 * direct + 0.28 * overlap)

    def _hard_match(self, product: Any) -> bool:
        return self._product_relevance(product) >= 0.48 and self._price_fit(product) > 0.0 and int(self.inventory.get(str(_get(product, "product_id", _get(product, "asin", ""))), 0)) > 0

    def _soft_satisfaction(self, product: Any) -> float:
        category = str(_get(product, "category", ""))
        category_score = self._latent.short_term_interest.get(category, 0.3)
        preferred_styles = set(_strings(_get(self.persona, "style_preferences", _get(self.persona, "styles", ()))))
        product_styles = set(_strings(_get(product, "styles", ()))) | set(_strings(_get(product, "attributes", _get(product, "attribute", ()))))
        style_score = len(preferred_styles & product_styles) / max(len(preferred_styles), 1)
        raw_rating = _get(product, "rating", 4.0)
        rating = (4.0 if raw_rating is None else float(raw_rating)) / 5.0
        return _bounded(0.48 * category_score + 0.22 * style_score + 0.30 * rating)

    def _price_fit(self, product: Any) -> float:
        price = max(float(_get(product, "price", 0.0)), 0.0)
        if price > self._latent.budget_remaining or self._latent.budget_remaining <= 0:
            return 0.0
        fraction = price / max(self._latent.budget_remaining, 1.0)
        return _bounded(1.0 - self._latent.price_sensitivity * fraction)

    def _strategy_fit(self, action: Mapping[str, Any], product: Any) -> float:
        strategy = str(action.get("strategy", "none"))
        if strategy in {"coupon", "discount", "low_price_alternative", "cheaper_alternative"}:
            return _bounded(0.45 + 0.55 * self._latent.price_sensitivity)
        if strategy == "review_summary":
            return (
                0.85
                if _strings(_get(product, "risk_tags", _get(product, "tags", ())))
                else 0.55
            )
        if strategy in {"price_comparison", "bundle"}:
            return 0.7
        return 0.5

    def _readiness(self, relevance: float) -> float:
        return _bounded(
            0.42 * self._latent.purchase_intent
            + 0.30 * relevance
            + 0.20 * self._latent.trust
            - 0.28 * self._latent.fatigue
            + 0.12
        )

    def _are_complements(self, first_id: str, second_id: str) -> bool:
        first = self._catalog_get(first_id)
        second = self._catalog_get(second_id)
        complement_ids = set(_strings(_get(first, "complement_product_ids", ())))
        if second_id in complement_ids:
            return True
        if hasattr(self.catalog, "complements"):
            try:
                inferred = self.catalog.complements(first_id, limit=16)
            except (KeyError, TypeError, ValueError):
                inferred = ()
            if second_id in {
                str(_get(product, "product_id", _get(product, "asin", "")))
                for product in inferred
            }:
                return True
        first_complements = set(_strings(_get(first, "complementary_categories", ())))
        second_category = str(_get(second, "category", ""))
        return second_category in first_complements

    def _coef(self, name: str, default: float) -> float:
        if self.calibration is None:
            return default
        coefficients = _get(self.calibration, "coefficients", {}) or {}
        if isinstance(coefficients, Mapping) and name in coefficients:
            raw = coefficients[name]
        elif _get(self.calibration, name, None) is not None:
            raw = _get(self.calibration, name)
        else:
            # Aggregate long-term datasets usually provide conditional event
            # rates, not logistic coefficients.  Translate a rate change into
            # an intercept delta while preserving the documented defaults.
            probability_field = {
                "like_bias": ("like_probability", 0.08),
                "click_bias": ("click_probability", 0.05),
                "cart_bias": ("cart_probability", 0.12),
                "buy_bias": ("purchase_probability", 0.28),
            }.get(name)
            if probability_field is None:
                return default
            field_name, prior = probability_field
            calibrated = self._calibration_value(field_name, prior)
            raw = default + self._logit(calibrated) - self._logit(prior)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def _calibration_value(self, name: str, default: float) -> float:
        if self.calibration is None:
            return float(default)
        raw = _get(self.calibration, name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    @staticmethod
    def _logit(probability: float) -> float:
        probability = _bounded(probability, 1e-6, 1.0 - 1e-6)
        return math.log(probability / (1.0 - probability))

    def _catalog_get(self, product_id: str) -> Any:
        if hasattr(self.catalog, "get"):
            product = self.catalog.get(str(product_id))
        elif isinstance(self.catalog, Mapping):
            product = self.catalog.get(str(product_id))
        else:
            product = None
        if product is None:
            raise KeyError(f"unknown product_id: {product_id}")
        return product

    def _require_visible_product(self, raw_product_id: Any) -> str:
        product_id = str(raw_product_id or "")
        if product_id not in self._visible_product_ids:
            raise FeedActionGuardError("product_not_visible")
        return product_id

    def _product_public(self, product: Any, *, detailed: bool) -> dict[str, Any]:
        product_id = str(_get(product, "product_id", _get(product, "asin", "")))
        row = {
            "product_id": product_id,
            "title": _clipped_text(_get(product, "title", ""), 300),
            "category": _clipped_text(_get(product, "category", ""), 160),
            "price": float(_get(product, "price", 0.0)),
            "rating": (
                0.0
                if _get(product, "rating", 0.0) is None
                else float(_get(product, "rating", 0.0))
            ),
            "popularity": float(_get(product, "popularity", 0.0)),
            "stock": int(self.inventory.get(product_id, 0)),
        }
        if detailed:
            row["attributes"] = _clipped_strings(
                _get(product, "attributes", _get(product, "attribute", ())),
                maximum_items=24,
                maximum_chars=160,
            )
        return row

    def _register_evidence(self, evidence_id: str, value: Any) -> None:
        self._evidence[str(evidence_id)] = {"value": _plain(value), "step": self.step_index}

    def _build_base_evidence(self) -> None:
        video = self.current_video
        if video is None:
            return
        public_video = self._public_video(video)
        video_id = evidence_component(public_video["video_id"])
        self._register_evidence(
            f"video.{video_id}.caption", public_video["caption"]
        )
        for index, value in enumerate(public_video["objects"]):
            self._register_evidence(f"video.{video_id}.object.{index:02d}", value)
        for index, value in enumerate(public_video["style"]):
            self._register_evidence(f"video.{video_id}.style.{index:02d}", value)
        for index, event in enumerate(self.history[-8:]):
            self._register_evidence(f"history.{event.get('event_id', index)}", event)
        self._register_evidence("persona.budget", _get(self.persona, "budget", _get(self.persona, "budget_limit", 300.0)))

    def _event(
        self,
        event_type: str,
        *,
        product_id: str | None = None,
        source_step: int | None = None,
        value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = f"e{self.step_index:03d}-{self._event_counter:06d}-{event_type}"
        self._event_counter += 1
        return {
            "event_id": event_id,
            "event_type": str(event_type),
            "step": self.step_index,
            "source_step": self.step_index if source_step is None else int(source_step),
            "product_id": product_id,
            "value": value,
            "metadata": dict(metadata or {}),
        }

    @staticmethod
    def _public_event(event: Mapping[str, Any]) -> dict[str, Any]:
        """Project an evaluator event without counterfactual or latent labels."""
        raw_metadata = event.get("metadata") or {}
        public_metadata = {
            key: raw_metadata[key]
            for key in ("video_id", "realized_at_step", "complementary")
            if key in raw_metadata
        }
        return {
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "step": event.get("step"),
            "source_step": event.get("source_step"),
            "product_id": event.get("product_id"),
            "value": event.get("value"),
            "metadata": public_metadata,
        }

    @staticmethod
    def _public_video(video: Any) -> dict[str, Any]:
        """Project video features without answer keys or simulator-only metadata."""
        raw_duration = _get(video, "duration_seconds", _get(video, "duration", None))
        return {
            "video_id": _identifier(_get(video, "video_id", ""), name="video_id"),
            "caption": _clipped_text(_get(video, "caption", ""), 600),
            "scene": _clipped_strings(
                _get(video, "scene", _get(video, "scenes", ())),
                maximum_items=32,
                maximum_chars=160,
            ),
            "objects": _clipped_strings(
                _get(video, "objects", ()), maximum_items=32, maximum_chars=160
            ),
            "style": _clipped_strings(
                _get(video, "style", _get(video, "styles", ())),
                maximum_items=32,
                maximum_chars=160,
            ),
            "topics": _clipped_strings(
                _get(video, "topics", ()), maximum_items=32, maximum_chars=160
            ),
            "asr": _clipped_text(_get(video, "asr", ""), 2000),
            "ocr": _clipped_text(_get(video, "ocr", ""), 1000),
            "creator_type": _clipped_text(_get(video, "creator_type", ""), 160),
            "duration_seconds": (
                None if raw_duration is None else float(raw_duration)
            ),
        }

    def _public_category_interests(self) -> Any:
        raw = _get(
            self.persona,
            "category_interests",
            _get(self.persona, "category_interest", ()),
        )
        if isinstance(raw, Mapping):
            result: dict[str, float] = {}
            for key, value in list(raw.items())[:32]:
                clipped = _clipped_text(key, 160)
                if clipped:
                    result[clipped] = float(value)
            return result
        return _clipped_strings(raw, maximum_items=32, maximum_chars=160)

    @staticmethod
    def _empty_reward() -> dict[str, float]:
        return {
            "qualified_purchase_value": 0.0,
            "satisfaction": 0.0,
            "purchase_satisfaction": 0.0,
            "correct_no_recommend": 0.0,
            "bundle_value": 0.0,
            "dwell_shaping": 0.0,
            "click_shaping": 0.0,
            "cart_shaping": 0.0,
            "interruption": 0.0,
            "irrelevant_recommendation": 0.0,
            "repeat_exposure": 0.0,
            "coupon_cost": 0.0,
            "return_penalty": 0.0,
            "unsupported_claim": 0.0,
            "total": 0.0,
        }
