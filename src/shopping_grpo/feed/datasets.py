"""Reproducible dataset factory for the long-horizon Feed profile.

This module deliberately contains no model client.  It converts public product truth
into split-isolated episode seeds, executes five deterministic/seeded behaviour
policies against :class:`~shopping_grpo.feed.simulator.FeedShoppingEnv`, and writes the
five artifacts used by the CPU MVP and later SFT/online-RL jobs.

Candidate-role annotations live in ``EpisodeSeed.metadata``.  They are evaluator data
and are never copied into model-visible observations or prompts.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.credit import counterfactual_advantage
from shopping_grpo.feed.evidence import evidence_component
from shopping_grpo.feed.manifest import (
    FEED_PROFILE_VERSIONS,
    audit_split_isolation,
    build_manifest,
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_rows,
    write_json_atomic,
)
from shopping_grpo.feed.observation import render_feed_observation
from shopping_grpo.feed.schema import EpisodeSeed, FeedAction, Persona, Product, Video, write_jsonl
from shopping_grpo.feed.simulator import FeedShoppingEnv
from shopping_grpo.feed.tools import FEED_TOOL_SCHEMAS
from shopping_grpo.feed.verl_adapter import build_public_tool_payload


DATASET_VERSION = "feed-datasets-v1"
MIN_FEED_LENGTH = 24
MAX_FEED_LENGTH = 48
SFT_MESSAGE_CHAR_BUDGET = 240_000
SFT_SHORT_WINDOW_MAX_STEPS = 8
POLICY_NAMES = ("Random", "Popular", "Similarity", "Rule", "Teacher")
CURRICULUM_STAGES = (
    "A_action_contract",
    "B_short_window",
    "C_long_horizon",
)
SPLIT_NAMES = ("train", "validation", "test")
ARTIFACT_PATHS = {
    "seeds": "seeds",
    "mixed_policy_logs": "mixed_policy_logs",
    "sft_trajectories": "sft_trajectories",
    "preference_pairs": "preference_pairs",
    "online_rl_tasks": "online_rl_tasks",
}

FEED_SYSTEM_PROMPT = """你是长程短视频 Feed 中的购物 Agent。每条视频最多调用三个信息工具，随后必须调用 commit_recommendation 提交一次 When–What–How 决策。只能使用 observation 和工具返回的公开证据；不得猜测用户隐藏意图、信任、疲劳或转化概率。工具回复采用 feed-tool-delta-v1：result 是本次公开结果，state_delta 仅更新列出的字段，未列出的公开状态继续沿用前文；每个窗口首条 observation 是完整检查点。购买不会结束 Feed，已经购买的商品不得重复营销；证据不足或干扰风险较高时选择 delay 或 no_recommend。"""

_PUBLIC_VIDEO_FIELDS = (
    "video_id",
    "caption",
    "scene",
    "objects",
    "style",
    "topics",
    "asr",
    "ocr",
    "creator_type",
    "duration_seconds",
)
_PUBLIC_OBSERVATION_FIELDS = (
    "observation_version",
    "environment_version",
    "episode_id",
    "step",
    "total_steps",
    "persona",
    "current_video",
    "recent_events",
    "cart",
    "purchased_product_ids",
    "visible_product_ids",
    "evidence_ids",
    "info_tool_calls",
    "max_info_tool_calls",
    "done",
)
_PUBLIC_EVENT_METADATA = ("video_id", "realized_at_step", "complementary")


def _stable_digest(seed: int, *parts: object) -> bytes:
    payload = ":".join((str(int(seed)), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _calibration_payload(calibration: Any) -> dict[str, Any] | None:
    if calibration is None:
        return None
    raw = calibration.to_dict() if hasattr(calibration, "to_dict") else calibration
    if not isinstance(raw, Mapping):
        raise TypeError("calibration must be a mapping or expose to_dict()")
    return json.loads(canonical_json(raw))


def _stable_integer(seed: int, *parts: object, modulo: int | None = None) -> int:
    value = int.from_bytes(_stable_digest(seed, *parts)[:8], "big")
    return value if modulo is None else value % modulo


def _stable_order(
    values: Iterable[Any], seed: int, *parts: object, identifier=lambda value: str(value)
) -> list[Any]:
    return sorted(
        values,
        key=lambda value: (
            _stable_digest(seed, *parts, identifier(value)),
            identifier(value),
        ),
    )


def _category(product: Product) -> str:
    return product.category.strip() or product.domain.strip() or "uncategorized"


def _attribute_set(product: Product) -> set[str]:
    return {
        item.strip().casefold()
        for item in (*product.attributes, *product.tags)
        if item.strip()
    }


@dataclass(frozen=True)
class _CandidateSet:
    anchor: Product
    hard_negative: Product
    cheaper_alternative: Product
    complement: Product
    unrelated: Product
    complement_source: str

    def role_ids(self) -> dict[str, list[str]]:
        return {
            "strong_relevant": [self.anchor.product_id],
            "hard_negative": [self.hard_negative.product_id],
            "cheaper_alternative": [self.cheaper_alternative.product_id],
            "complement": [self.complement.product_id],
            "unrelated": [self.unrelated.product_id],
        }


class _CatalogIndex:
    """Small deterministic indexes used only during seed construction."""

    def __init__(self, catalog: ProductCatalog) -> None:
        self.catalog = catalog
        groups: dict[str, list[Product]] = defaultdict(list)
        attribute_products: dict[str, list[Product]] = defaultdict(list)
        for product in catalog:
            groups[_category(product)].append(product)
            for attribute in _attribute_set(product):
                attribute_products[attribute].append(product)
        self.groups = {
            category: tuple(sorted(products, key=lambda item: item.product_id))
            for category, products in groups.items()
        }
        # A category needs distinct anchor/cheaper/hard-negative products and price
        # variation for the role labels to be truthful.
        self.eligible_categories = tuple(
            sorted(
                category
                for category, products in self.groups.items()
                if len(products) >= 3
                and min(product.price for product in products)
                < max(product.price for product in products)
            )
        )
        self.attribute_products = {
            attribute: tuple(sorted(products, key=lambda item: item.product_id))
            for attribute, products in attribute_products.items()
        }
        self.all_products = tuple(sorted(catalog, key=lambda item: item.product_id))
        if len(self.eligible_categories) < 3:
            raise ValueError(
                "catalog needs at least three categories with three products and price variation"
            )

    def categories_for_episode(
        self, *, seed: int, episode_id: str, feed_length: int
    ) -> tuple[str, ...]:
        maximum = min(5, feed_length, len(self.eligible_categories))
        count = 3 + _stable_integer(seed, episode_id, "category_count", modulo=maximum - 2)
        ordered = _stable_order(
            self.eligible_categories,
            seed,
            episode_id,
            "categories",
        )
        return tuple(ordered[:count])

    def candidates(
        self,
        *,
        category: str,
        seed: int,
        episode_id: str,
        step: int,
    ) -> _CandidateSet:
        products = self.groups[category]
        lowest_price = min(product.price for product in products)
        anchor_pool = [product for product in products if product.price > lowest_price]
        anchor = _stable_order(
            anchor_pool,
            seed,
            episode_id,
            step,
            "anchor",
            identifier=lambda item: item.product_id,
        )[0]

        cheaper_pool = [product for product in products if product.price < anchor.price]
        cheaper = sorted(
            cheaper_pool,
            key=lambda item: (item.price, item.product_id),
        )[0]

        anchor_attributes = _attribute_set(anchor)
        hard_pool = [
            product
            for product in products
            if product.product_id not in {anchor.product_id, cheaper.product_id}
        ]
        hard_negative = sorted(
            hard_pool,
            key=lambda item: (
                len(anchor_attributes & _attribute_set(item)),
                -item.price,
                item.product_id,
            ),
        )[0]

        complement, complement_source = self._complement(
            anchor,
            excluded={anchor.product_id, cheaper.product_id, hard_negative.product_id},
            seed=seed,
            episode_id=episode_id,
            step=step,
        )
        unrelated = self._unrelated(
            anchor,
            excluded={
                anchor.product_id,
                cheaper.product_id,
                hard_negative.product_id,
                complement.product_id,
            },
            seed=seed,
            episode_id=episode_id,
            step=step,
        )
        return _CandidateSet(
            anchor=anchor,
            hard_negative=hard_negative,
            cheaper_alternative=cheaper,
            complement=complement,
            unrelated=unrelated,
            complement_source=complement_source,
        )

    def _complement(
        self,
        anchor: Product,
        *,
        excluded: set[str],
        seed: int,
        episode_id: str,
        step: int,
    ) -> tuple[Product, str]:
        explicit = [
            self.catalog.get(product_id)
            for product_id in anchor.complement_product_ids
            if product_id not in excluded
        ]
        explicit = [product for product in explicit if product is not None]
        if explicit:
            return (
                _stable_order(
                    explicit,
                    seed,
                    episode_id,
                    step,
                    "explicit_complement",
                    identifier=lambda item: item.product_id,
                )[0],
                "catalog_explicit",
            )

        shared: dict[str, Product] = {}
        for attribute in sorted(_attribute_set(anchor)):
            for product in self.attribute_products.get(attribute, ()):
                if (
                    product.product_id not in excluded
                    and _category(product) != _category(anchor)
                ):
                    shared[product.product_id] = product
        if shared:
            ranked = sorted(
                shared.values(),
                key=lambda item: (
                    -len(_attribute_set(anchor) & _attribute_set(item)),
                    _stable_digest(seed, episode_id, step, "shared_complement", item.product_id),
                    item.product_id,
                ),
            )
            return ranked[0], "shared_public_attribute"

        # ShopSimulator does not contain a universal complement graph.  The final
        # fallback is explicitly labelled as a cross-category heuristic rather than
        # silently turning it into ground truth.
        candidate = self._cyclic_product(
            seed=seed,
            parts=(episode_id, step, "complement_heuristic"),
            predicate=lambda item: (
                item.product_id not in excluded and _category(item) != _category(anchor)
            ),
        )
        return candidate, "cross_category_heuristic"

    def _unrelated(
        self,
        anchor: Product,
        *,
        excluded: set[str],
        seed: int,
        episode_id: str,
        step: int,
    ) -> Product:
        anchor_attributes = _attribute_set(anchor)
        return self._cyclic_product(
            seed=seed,
            parts=(episode_id, step, "unrelated"),
            predicate=lambda item: (
                item.product_id not in excluded
                and _category(item) != _category(anchor)
                and not (anchor_attributes & _attribute_set(item))
            ),
            fallback_predicate=lambda item: (
                item.product_id not in excluded and _category(item) != _category(anchor)
            ),
        )

    def _cyclic_product(
        self,
        *,
        seed: int,
        parts: Sequence[object],
        predicate,
        fallback_predicate=None,
    ) -> Product:
        start = _stable_integer(seed, *parts, modulo=len(self.all_products))
        ordered = self.all_products[start:] + self.all_products[:start]
        for product in ordered:
            if predicate(product):
                return product
        if fallback_predicate is not None:
            for product in ordered:
                if fallback_predicate(product):
                    return product
        raise ValueError("catalog cannot provide the requested distinct candidate role")


def _split_counts(episodes: int) -> dict[str, int]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 3:
        raise ValueError("episodes must be an integer of at least 3")
    validation = max(1, round(episodes * 0.1))
    test = max(1, round(episodes * 0.1))
    while validation + test >= episodes:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            break
    return {
        "train": episodes - validation - test,
        "validation": validation,
        "test": test,
    }


def _validate_feed_length(feed_length: int) -> None:
    if (
        isinstance(feed_length, bool)
        or not isinstance(feed_length, int)
        or not MIN_FEED_LENGTH <= feed_length <= MAX_FEED_LENGTH
    ):
        raise ValueError(
            f"feed_length must be an integer from {MIN_FEED_LENGTH} to {MAX_FEED_LENGTH}"
        )


def generate_episode_splits(
    catalog: ProductCatalog,
    *,
    episodes: int,
    feed_length: int = 24,
    seed: int = 42,
) -> dict[str, list[EpisodeSeed]]:
    """Create deterministic train/validation/test seeds with disjoint IDs.

    Every episode contains three to five categories and, for every video, distinct
    strong, hard-negative, cheaper, complement and unrelated candidates.
    """

    if not isinstance(catalog, ProductCatalog):
        raise TypeError("catalog must be a ProductCatalog")
    _validate_feed_length(feed_length)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    index = _CatalogIndex(catalog)
    counts = _split_counts(episodes)
    result: dict[str, list[EpisodeSeed]] = {name: [] for name in SPLIT_NAMES}
    global_index = 0
    for split in SPLIT_NAMES:
        for split_index in range(counts[split]):
            episode_id = f"feed-{split}-{split_index:06d}"
            persona_id = f"persona-{split}-{split_index:06d}"
            episode_seed_value = _stable_integer(
                seed, split, split_index, "episode_seed", modulo=2**31 - 1
            )
            categories = index.categories_for_episode(
                seed=seed,
                episode_id=episode_id,
                feed_length=feed_length,
            )
            videos: list[Video] = []
            candidate_roles: dict[str, dict[str, list[str]]] = {}
            candidate_role_sources: dict[str, dict[str, str]] = {}
            product_ids: set[str] = set()
            anchor_prices: list[float] = []
            persona_categories: set[str] = set()
            persona_styles: set[str] = set()
            for step in range(feed_length):
                category = categories[step % len(categories)]
                candidates = index.candidates(
                    category=category,
                    seed=seed,
                    episode_id=episode_id,
                    step=step,
                )
                anchor = candidates.anchor
                video_id = f"{episode_id}-video-{step:03d}"
                roles = candidates.role_ids()
                candidate_roles[video_id] = roles
                candidate_role_sources[video_id] = {
                    "strong_relevant": "video_related_product",
                    "hard_negative": "same_category_low_attribute_overlap",
                    "cheaper_alternative": "same_category_strictly_lower_price",
                    "complement": candidates.complement_source,
                    "unrelated": "different_category_no_shared_attribute",
                }
                for identifiers in roles.values():
                    product_ids.update(identifiers)
                attributes = tuple(anchor.attributes[:4])
                style = tuple(anchor.attributes[:2]) or ("product_demo",)
                videos.append(
                    Video(
                        video_id=video_id,
                        caption=f"{anchor.title}｜{anchor.category}",
                        scene=("short_video_feed",),
                        objects=attributes or (anchor.title,),
                        style=style,
                        topics=(category,),
                        asr=anchor.title,
                        ocr=f"¥{anchor.price:.2f}",
                        creator_type=("merchant" if step % 3 == 0 else "creator"),
                        related_product_ids=(anchor.product_id,),
                        duration_seconds=float(
                            18 + _stable_integer(seed, episode_id, step, "duration", modulo=43)
                        ),
                    )
                )
                anchor_prices.append(anchor.price)
                persona_categories.add(anchor.category)
                persona_styles.update(style)

            sorted_prices = sorted(anchor_prices)
            median_price = sorted_prices[len(sorted_prices) // 2]
            budget = round(max(100.0, median_price * 3.0), 2)
            persona = Persona(
                persona_id=persona_id,
                country="CN",
                budget=budget,
                category_interests=tuple(sorted(persona_categories)),
                style_preferences=tuple(sorted(persona_styles)[:8]),
                price_sensitivity=round(
                    0.25
                    + 0.65
                    * (
                        _stable_integer(seed, episode_id, "price_sensitivity", modulo=1000)
                        / 999.0
                    ),
                    6,
                ),
                tags=("synthetic_feed_persona",),
                metadata={"source": "deterministic_product_truth"},
            )
            ordered_product_ids = tuple(sorted(product_ids))
            inventory = tuple(
                (
                    product_id,
                    1
                    + _stable_integer(
                        seed, episode_id, product_id, "inventory", modulo=5
                    ),
                )
                for product_id in ordered_product_ids
            )
            breaks = (feed_length // 2,) if feed_length >= 6 else ()
            result[split].append(
                EpisodeSeed(
                    episode_id=episode_id,
                    persona=persona,
                    videos=tuple(videos),
                    product_ids=ordered_product_ids,
                    seed=episode_seed_value,
                    inventory=inventory,
                    session_breaks=breaks,
                    metadata={
                        "dataset_version": DATASET_VERSION,
                        "split": split,
                        "global_index": global_index,
                        "categories": list(categories),
                        "candidate_roles_by_video": candidate_roles,
                        "candidate_role_sources_by_video": candidate_role_sources,
                    },
                )
            )
            global_index += 1
    audit_split_isolation(result)
    return result


def _safe_event(event: Mapping[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "step": event.get("step"),
        "source_step": event.get("source_step"),
        "product_id": event.get("product_id"),
        "value": event.get("value"),
        "metadata": {
            key: metadata[key] for key in _PUBLIC_EVENT_METADATA if key in metadata
        },
    }


def _safe_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Defense-in-depth projection used for prompts and exported policy context."""

    result = {
        key: deepcopy(observation[key])
        for key in _PUBLIC_OBSERVATION_FIELDS
        if key in observation
    }
    persona = result.get("persona")
    if isinstance(persona, Mapping):
        result["persona"] = {
            key: deepcopy(persona[key])
            for key in ("persona_id", "budget", "category_interests", "style_preferences")
            if key in persona
        }
    video = result.get("current_video")
    if isinstance(video, Mapping):
        result["current_video"] = {
            key: deepcopy(video[key]) for key in _PUBLIC_VIDEO_FIELDS if key in video
        }
    events = result.get("recent_events")
    if isinstance(events, list):
        result["recent_events"] = [
            _safe_event(event) for event in events if isinstance(event, Mapping)
        ]
    # The canonical renderer is deliberately fail-closed and therefore doubles as a
    # generation-time leakage assertion.
    render_feed_observation(result)
    return result


def _action_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, FeedAction):
        return action.to_dict()
    if hasattr(action, "to_dict"):
        action = action.to_dict()
    if not isinstance(action, Mapping):
        raise TypeError("policy action must be a mapping or FeedAction")
    return FeedAction.from_dict(action).to_dict()


class _FallbackPolicy:
    """Transparent standard-library fallback used only when policies are unavailable."""

    def __init__(self, name: str, *, seed: int) -> None:
        self.name = name
        self.seed = seed

    def act(self, env: FeedShoppingEnv) -> dict[str, Any]:
        observation = _safe_observation(env.observation())
        step = int(observation["step"])
        video = observation.get("current_video") or {}
        query_parts = [
            str(video.get("caption", "")),
            *(str(item) for item in video.get("topics", [])[:2]),
            *(str(item) for item in video.get("objects", [])[:3]),
        ]
        query = " ".join(part for part in query_parts if part).strip()[:300]
        retrieval = env.call_tool("retrieve_products", {"query": query or "商品"})
        candidates = list(retrieval.get("products") or [])
        purchased = set(observation.get("purchased_product_ids") or [])
        candidates = [row for row in candidates if row.get("product_id") not in purchased]

        decision = self._decision(observation, bool(candidates))
        if decision != "recommend" or not candidates:
            return {
                "decision": decision if candidates else "no_recommend",
                "surface": "none",
                "strategy": "none",
                "relationship": "primary",
                "product_ids": [],
                "evidence_ids": [],
                "explanation": "当前公开证据不足或干扰风险较高。",
            }

        selected = self._select(candidates, observation)
        product_id = str(selected["product_id"])
        current = env.observation()
        evidence = [
            item
            for item in current.get("evidence_ids", [])
            if item == f"product.{evidence_component(product_id)}.retrieved"
        ]
        context = [
            item
            for item in current.get("evidence_ids", [])
            if str(item).startswith(("video.", "history.", "persona."))
        ]
        return {
            "decision": "recommend",
            "surface": "product_card",
            "strategy": "direct",
            "relationship": "primary",
            "product_ids": [product_id],
            "evidence_ids": [*evidence[:1], *context[:1]],
            "explanation": "基于当前视频内容与已检索商品事实进行推荐。",
        }

    def _decision(self, observation: Mapping[str, Any], has_candidates: bool) -> str:
        if not has_candidates:
            return "no_recommend"
        step = int(observation["step"])
        if self.name == "Random":
            return ("recommend", "delay", "no_recommend")[
                _stable_integer(
                    self.seed,
                    observation["episode_id"],
                    step,
                    "random_decision",
                    modulo=3,
                )
            ]
        if self.name == "Rule":
            return "recommend" if step % 3 == 2 else "delay"
        if self.name == "Teacher":
            recent = observation.get("recent_events") or []
            recent_interventions = sum(
                event.get("event_type") == "intervention" for event in recent[-6:]
            )
            return "no_recommend" if recent_interventions >= 3 else "recommend"
        if self.name == "Popular" and step % 4 == 3:
            return "delay"
        return "recommend"

    def _select(
        self, candidates: list[Mapping[str, Any]], observation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self.name == "Random":
            index = _stable_integer(
                self.seed,
                observation["episode_id"],
                observation["step"],
                "random_product",
                modulo=len(candidates),
            )
            return candidates[index]
        if self.name == "Popular":
            # Product popularity is intentionally not present in tool results.  The
            # fallback uses rating as an auditable public proxy and labels itself.
            return sorted(
                candidates,
                key=lambda row: (-float(row.get("rating") or 0.0), str(row["product_id"])),
            )[0]
        if self.name in {"Rule", "Teacher"}:
            budget = float((observation.get("persona") or {}).get("budget") or 0.0)
            return sorted(
                candidates,
                key=lambda row: (
                    float(row.get("price") or 0.0) > budget,
                    -float(row.get("rating") or 0.0),
                    float(row.get("price") or 0.0),
                    str(row["product_id"]),
                ),
            )[0]
        return candidates[0]


def _external_policy(name: str, *, seed: int):
    try:
        from shopping_grpo.feed import policies as policy_module
    except ImportError:
        return None
    registry = getattr(policy_module, "POLICY_REGISTRY", {})
    candidate = None
    if isinstance(registry, Mapping):
        for key in (name, name.casefold(), f"{name}Policy", f"{name.casefold()}_policy"):
            if key in registry:
                candidate = registry[key]
                break
    if candidate is None:
        candidate = getattr(policy_module, f"{name}Policy", None)
    if candidate is None:
        return None
    if hasattr(candidate, "act") and not isinstance(candidate, type):
        return candidate
    for kwargs in ({"seed": seed}, {"random_seed": seed}, {}):
        try:
            policy = candidate(**kwargs)
        except TypeError:
            continue
        if hasattr(policy, "act"):
            return policy
    return None


def _tool_message(
    *,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": canonical_json(arguments),
                },
            }
        ],
    }
    reply = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": canonical_json(result),
    }
    return assistant, reply


def _public_product_ids(value: Any) -> set[str]:
    """Collect product IDs exposed by one public information-tool result."""

    identifiers: set[str] = set()
    if isinstance(value, Mapping):
        product_id = value.get("product_id")
        if isinstance(product_id, str) and product_id:
            identifiers.add(product_id)
        for nested in value.values():
            identifiers.update(_public_product_ids(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            identifiers.update(_public_product_ids(nested))
    return identifiers


def _advance_public_information_state(
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild an information-call state delta using public results only."""

    evolved = deepcopy(dict(observation))
    current_calls = int(evolved.get("info_tool_calls") or 0)
    maximum_calls = int(evolved.get("max_info_tool_calls") or current_calls + 1)
    evolved["info_tool_calls"] = min(current_calls + 1, maximum_calls)
    evolved["visible_product_ids"] = sorted(
        {
            *(str(item) for item in evolved.get("visible_product_ids", ())),
            *_public_product_ids(result),
        }
    )
    raw_evidence = result.get("evidence_ids", ())
    evidence = (
        raw_evidence
        if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes))
        else ()
    )
    evolved["evidence_ids"] = sorted(
        {
            *(str(item) for item in evolved.get("evidence_ids", ())),
            *(str(item) for item in evidence),
        }
    )
    return evolved


def _rollout_once(
    seed_record: EpisodeSeed,
    catalog: ProductCatalog,
    *,
    policy_name: str,
    generation_seed: int,
    use_external_policy: bool,
    calibration: Any = None,
    override_step: int | None = None,
    override_action: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    env = FeedShoppingEnv(seed_record, catalog, calibration=calibration)
    policy = (
        _external_policy(policy_name, seed=generation_seed)
        if use_external_policy
        else None
    )
    policy_source = (
        f"shopping_grpo.feed.policies.{type(policy).__name__}"
        if policy is not None
        else "shopping_grpo.feed.datasets._FallbackPolicy"
    )
    policy = policy or _FallbackPolicy(policy_name, seed=generation_seed)
    transitions: list[dict[str, Any]] = []
    message_chunks: list[list[dict[str, Any]]] = []

    while not env.done:
        step = env.step_index
        before_observation = _safe_observation(env.observation())
        before_records = len(env.tool_records)
        proposed = policy.act(env)
        action = _action_dict(proposed)
        if override_step == step and override_action is not None:
            action = _action_dict(override_action)
        information_records = [
            deepcopy(record) for record in env.tool_records[before_records:]
        ]
        result = env.step(action)
        after_observation = _safe_observation(result.observation)
        public_events = [
            _safe_event(event) for event in result.events if isinstance(event, Mapping)
        ]
        transitions.append(
            {
                "step": step,
                "pre_observation": before_observation,
                "tool_records": information_records,
                "action": action,
                "events": public_events,
                "reward": deepcopy(result.reward_breakdown),
                "post_observation": after_observation,
                "done": result.done,
            }
        )

        chunk: list[dict[str, Any]] = [
            {"role": "user", "content": render_feed_observation(before_observation)}
        ]
        public_tool_state = deepcopy(before_observation)
        for tool_index, record in enumerate(information_records):
            call_id = f"call-{seed_record.episode_id}-{policy_name.lower()}-{step:03d}-{tool_index:02d}"
            tool_name = str(record["tool_name"])
            tool_result = dict(record.get("result") or {})
            public_tool_state = _advance_public_information_state(
                public_tool_state,
                tool_result,
            )
            assistant, reply = _tool_message(
                call_id=call_id,
                name=tool_name,
                arguments=dict(record.get("arguments") or {}),
                result=build_public_tool_payload(
                    tool_name,
                    tool_result,
                    public_tool_state,
                ),
            )
            chunk.extend((assistant, reply))
        commit_id = f"call-{seed_record.episode_id}-{policy_name.lower()}-{step:03d}-commit"
        assistant, reply = _tool_message(
            call_id=commit_id,
            name="commit_recommendation",
            arguments=action,
            result=build_public_tool_payload(
                "commit_recommendation",
                {"events": public_events, "done": result.done},
                after_observation,
            ),
        )
        chunk.extend((assistant, reply))
        message_chunks.append(chunk)

    summary = env.summary()
    public_summary = {
        "episode_id": summary["episode_id"],
        "environment_version": summary["environment_version"],
        "reward_version": summary["reward_version"],
        "steps": summary["steps"],
        "done": summary["done"],
        "episode_return": summary["episode_return"],
        "net_revenue": summary["net_revenue"],
        "event_counts": summary["event_counts"],
    }
    row = {
        "artifact_version": DATASET_VERSION,
        "trajectory_id": f"{seed_record.episode_id}:{policy_name.casefold()}",
        "episode_id": seed_record.episode_id,
        "persona_id": seed_record.persona.persona_id,
        "split": seed_record.metadata["split"],
        "behavior_policy": policy_name,
        "policy_source": policy_source,
        "common_random_seed": seed_record.seed,
        "transitions": transitions,
        "summary": public_summary,
        # Offline evaluators need delayed-outcome labels that must never enter an
        # observation, SFT message, preference prompt, or online-RL prompt.
        "evaluator_summary": {
            **deepcopy(summary),
            "evaluator_only": True,
        },
    }
    return row, message_chunks


def _rollout(
    seed_record: EpisodeSeed,
    catalog: ProductCatalog,
    *,
    policy_name: str,
    generation_seed: int,
    prefer_external_policies: bool,
    calibration: Any = None,
    override_step: int | None = None,
    override_action: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    if prefer_external_policies:
        try:
            return _rollout_once(
                seed_record,
                catalog,
                policy_name=policy_name,
                generation_seed=generation_seed,
                use_external_policy=True,
                calibration=calibration,
                override_step=override_step,
                override_action=override_action,
            )
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            row, chunks = _rollout_once(
                seed_record,
                catalog,
                policy_name=policy_name,
                generation_seed=generation_seed,
                use_external_policy=False,
                calibration=calibration,
                override_step=override_step,
                override_action=override_action,
            )
            row["policy_source"] += f":after_{type(exc).__name__}"
            return row, chunks
    return _rollout_once(
        seed_record,
        catalog,
        policy_name=policy_name,
        generation_seed=generation_seed,
        use_external_policy=False,
        calibration=calibration,
        override_step=override_step,
        override_action=override_action,
    )


def _flatten_window_chunks(
    chunks: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep one full checkpoint, then advance only through tool deltas."""

    flattened: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        if not chunk or chunk[0].get("role") != "user":
            raise ValueError("every SFT step chunk must start with a user checkpoint")
        messages = chunk if chunk_index == 0 else chunk[1:]
        flattened.extend(deepcopy(dict(message)) for message in messages)
    return flattened


def _sft_messages(
    chunks: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": FEED_SYSTEM_PROMPT},
        *_flatten_window_chunks(chunks),
    ]


def _serialized_message_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    """Return the exact character count used by deterministic JSONL output."""

    return len(canonical_json(messages))


def _bounded_sft_prefix(
    chunks: Sequence[Sequence[Mapping[str, Any]]],
    *,
    max_steps: int,
) -> tuple[int, list[dict[str, Any]], int]:
    """Take the longest non-empty prefix that fits without splitting a step."""

    limit = min(len(chunks), max_steps)
    for end_step in range(limit, 0, -1):
        messages = _sft_messages(chunks[:end_step])
        serialized_chars = _serialized_message_chars(messages)
        if serialized_chars <= SFT_MESSAGE_CHAR_BUDGET:
            return end_step, messages, serialized_chars
    first_step_chars = _serialized_message_chars(_sft_messages(chunks[:1]))
    raise ValueError(
        "one atomic Feed step exceeds SFT_MESSAGE_CHAR_BUDGET: "
        f"{first_step_chars} > {SFT_MESSAGE_CHAR_BUDGET}"
    )


def _sft_row(
    seed_record: EpisodeSeed,
    *,
    stage: str,
    start_step: int,
    end_step: int,
    messages: list[dict[str, Any]],
    serialized_chars: int,
    window_index: int,
    window_count: int,
) -> dict[str, Any]:
    suffix = f":window-{window_index:03d}" if window_count > 1 else ""
    return {
        "artifact_version": DATASET_VERSION,
        "trajectory_id": f"{seed_record.episode_id}:teacher:{stage}{suffix}",
        "task_id": seed_record.episode_id,
        "episode_id": seed_record.episode_id,
        "persona_id": seed_record.persona.persona_id,
        "split": seed_record.metadata["split"],
        "curriculum_stage": stage,
        "window_index": window_index,
        "window_count": window_count,
        "window_start_step": start_step,
        "window_end_step_exclusive": end_step,
        "window_steps": end_step - start_step,
        "full_episode_steps": len(seed_record.videos),
        "serialized_message_chars": serialized_chars,
        "message_char_budget": SFT_MESSAGE_CHAR_BUDGET,
        "messages": messages,
        "tools": deepcopy(FEED_TOOL_SCHEMAS),
    }


def _sft_rows(
    seed_record: EpisodeSeed,
    chunks: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if len(chunks) != len(seed_record.videos):
        raise ValueError("SFT message chunks must cover every Feed video exactly once")

    action_end, action_messages, action_chars = _bounded_sft_prefix(chunks, max_steps=1)
    short_target = min(
        len(chunks),
        max(2, min(SFT_SHORT_WINDOW_MAX_STEPS, (len(chunks) + 2) // 3)),
    )
    short_end, short_messages, short_chars = _bounded_sft_prefix(
        chunks,
        max_steps=short_target,
    )
    long_messages = _sft_messages(chunks)
    long_chars = _serialized_message_chars(long_messages)
    if long_chars > SFT_MESSAGE_CHAR_BUDGET:
        raise ValueError(
            "full long-horizon SFT trajectory exceeds "
            f"SFT_MESSAGE_CHAR_BUDGET: {long_chars} > {SFT_MESSAGE_CHAR_BUDGET}"
        )

    rows = [
        _sft_row(
            seed_record,
            stage="A_action_contract",
            start_step=0,
            end_step=action_end,
            messages=action_messages,
            serialized_chars=action_chars,
            window_index=0,
            window_count=1,
        ),
        _sft_row(
            seed_record,
            stage="B_short_window",
            start_step=0,
            end_step=short_end,
            messages=short_messages,
            serialized_chars=short_chars,
            window_index=0,
            window_count=1,
        ),
    ]
    rows.append(
        _sft_row(
            seed_record,
            stage="C_long_horizon",
            start_step=0,
            end_step=len(chunks),
            messages=long_messages,
            serialized_chars=long_chars,
            window_index=0,
            window_count=1,
        )
    )
    return rows


def _assistant_commit(action: Mapping[str, Any], *, call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "commit_recommendation",
                    "arguments": canonical_json(action),
                },
            }
        ],
    }


def _preference_row(
    seed_record: EpisodeSeed,
    catalog: ProductCatalog,
    teacher_log: Mapping[str, Any],
    *,
    calibration: Any = None,
) -> dict[str, Any]:
    transitions = list(teacher_log["transitions"])
    recommend_steps = [
        int(item["step"])
        for item in transitions
        if item["action"]["decision"] == "recommend"
    ]
    target_step = recommend_steps[len(recommend_steps) // 2] if recommend_steps else len(transitions) // 2
    factual = transitions[target_step]["action"]
    if factual["decision"] == "recommend":
        counterfactual_action = {
            "decision": "no_recommend",
            "surface": "none",
            "strategy": "none",
            "relationship": "primary",
            "product_ids": [],
            "evidence_ids": [],
            "explanation": "反事实：本次不打断内容消费。",
        }
    else:
        counterfactual_action = {
            "decision": "delay",
            "surface": "none",
            "strategy": "none",
            "relationship": "primary",
            "product_ids": [],
            "evidence_ids": [],
            "explanation": "反事实：推迟本次干预。",
        }
    comparison = counterfactual_advantage(
        lambda: FeedShoppingEnv(seed_record, catalog, calibration=calibration),
        transitions,
        target_step,
        replacement_action=counterfactual_action,
    )
    factual_return = float(comparison["factual_return"])
    counter_return = float(comparison["counterfactual_return"])
    factual_is_chosen = factual_return >= counter_return
    chosen_action = factual if factual_is_chosen else counterfactual_action
    rejected_action = counterfactual_action if factual_is_chosen else factual
    chosen_return = factual_return if factual_is_chosen else counter_return
    rejected_return = counter_return if factual_is_chosen else factual_return
    target_transition = transitions[target_step]
    observation = target_transition["pre_observation"]
    pair_id = f"{seed_record.episode_id}:step-{target_step:03d}:crn"
    prompt = [
        {"role": "system", "content": FEED_SYSTEM_PROMPT},
        {"role": "user", "content": render_feed_observation(observation)},
    ]
    public_tool_state = deepcopy(observation)
    for tool_index, record in enumerate(target_transition["tool_records"]):
        tool_name = str(record["tool_name"])
        tool_result = dict(record.get("result") or {})
        public_tool_state = _advance_public_information_state(
            public_tool_state,
            tool_result,
        )
        assistant, reply = _tool_message(
            call_id=f"context-{pair_id}-{tool_index:02d}",
            name=tool_name,
            arguments=dict(record.get("arguments") or {}),
            result=build_public_tool_payload(
                tool_name,
                tool_result,
                public_tool_state,
            ),
        )
        prompt.extend((assistant, reply))
    return {
        "artifact_version": DATASET_VERSION,
        "pair_id": pair_id,
        "task_id": seed_record.episode_id,
        "episode_id": seed_record.episode_id,
        "persona_id": seed_record.persona.persona_id,
        "split": seed_record.metadata["split"],
        "target_step": target_step,
        "common_random_seed": seed_record.seed,
        "common_random_numbers": True,
        "prompt": prompt,
        "chosen": _assistant_commit(chosen_action, call_id=f"candidate-{pair_id}"),
        "rejected": _assistant_commit(rejected_action, call_id=f"candidate-{pair_id}"),
        "chosen_action": chosen_action,
        "rejected_action": rejected_action,
        "chosen_episode_return": chosen_return,
        "rejected_episode_return": rejected_return,
        "return_margin": chosen_return - rejected_return,
        "factual_action": factual,
        "counterfactual_action": counterfactual_action,
        "factual_episode_return": factual_return,
        "counterfactual_episode_return": counter_return,
        "counterfactual_method": comparison["method"],
        "counterfactual_policy_source": (
            "shopping_grpo.feed.credit.counterfactual_advantage:recorded_action_replay"
        ),
        "tools": deepcopy(FEED_TOOL_SCHEMAS),
    }


def _online_rl_row(
    seed_record: EpisodeSeed,
    catalog: ProductCatalog,
    *,
    calibration: Any = None,
) -> dict[str, Any]:
    initial = _safe_observation(
        FeedShoppingEnv(seed_record, catalog, calibration=calibration).observation()
    )
    catalog_products = [
        catalog.require(product_id).to_dict()
        for product_id in seed_record.product_ids
    ]
    row = {
        "data_source": "feed-shopping-v1",
        "task_id": seed_record.episode_id,
        "episode_id": seed_record.episode_id,
        "ability": "long_horizon_feed_shopping",
        "prompt": [
            {"role": "system", "content": FEED_SYSTEM_PROMPT},
            {"role": "user", "content": render_feed_observation(initial)},
        ],
        "tools": deepcopy(FEED_TOOL_SCHEMAS),
        "episode_seed": seed_record.to_dict(),
        "catalog_products": catalog_products,
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "episode_id": seed_record.episode_id,
                "reward_version": FEED_PROFILE_VERSIONS["reward"],
            },
        },
        "extra_info": {
            "episode_id": seed_record.episode_id,
            "task_id": seed_record.episode_id,
            "persona_id": seed_record.persona.persona_id,
            "split": seed_record.metadata["split"],
            "profile_versions": dict(FEED_PROFILE_VERSIONS),
            "episode_seed": seed_record.to_dict(),
        },
    }
    calibration_payload = _calibration_payload(calibration)
    if calibration_payload is not None:
        row["calibration"] = calibration_payload
    return row


def load_product_catalog(path: str | Path) -> ProductCatalog:
    """Load either a Feed product JSONL or the ShopSimulator JSON/JSON.GZ archive."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"catalog does not exist: {source}")
    if ".jsonl" in source.suffixes:
        return ProductCatalog.from_jsonl(source)
    return ProductCatalog.from_shopsimulator(source)


def _prepare_output(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass force=True to replace artifacts"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for relative in ARTIFACT_PATHS.values():
            path = output_dir / relative
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()


def _catalog_source(
    catalog: ProductCatalog,
    source_path: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "product_count": len(catalog),
        "product_rows_sha256": sha256_rows(
            product.to_dict() for product in catalog
        ),
    }
    if source_path is not None:
        resolved = source_path.resolve()
        repository_root = Path(__file__).resolve().parents[3]
        try:
            portable_path = resolved.relative_to(repository_root).as_posix()
        except ValueError:
            portable_path = resolved.name
        result.update(
            {
                "path": portable_path,
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    else:
        result["path"] = None
    return result


def generate_feed_artifacts(
    catalog: ProductCatalog | str | Path,
    output_dir: str | Path,
    *,
    episodes: int,
    feed_length: int = 24,
    seed: int = 42,
    force: bool = False,
    prefer_external_policies: bool = True,
    calibration: Any = None,
) -> dict[str, Any]:
    """Generate all five Feed artifacts and a hash-verifiable manifest.

    No LLM endpoint or training runtime is imported.  Returned manifests are suitable
    for :func:`shopping_grpo.feed.manifest.verify_manifest`.
    """

    _validate_feed_length(feed_length)
    source_path = None if isinstance(catalog, ProductCatalog) else Path(catalog)
    product_catalog = catalog if isinstance(catalog, ProductCatalog) else load_product_catalog(catalog)
    calibration_payload = _calibration_payload(calibration)
    target = Path(output_dir)
    _prepare_output(target, force=force)
    splits = generate_episode_splits(
        product_catalog,
        episodes=episodes,
        feed_length=feed_length,
        seed=seed,
    )
    split_summary = audit_split_isolation(splits)
    artifact_counts: dict[str, dict[str, int]] = {
        name: {} for name in ARTIFACT_PATHS
    }
    curriculum_counts: dict[str, dict[str, int]] = {
        split: {stage: 0 for stage in CURRICULUM_STAGES} for split in SPLIT_NAMES
    }
    policy_source_counts: dict[str, int] = defaultdict(int)

    for split in SPLIT_NAMES:
        seed_rows = [record.to_dict() for record in splits[split]]
        write_jsonl(target / ARTIFACT_PATHS["seeds"] / f"{split}.jsonl", seed_rows)
        artifact_counts["seeds"][split] = len(seed_rows)

        logs: list[dict[str, Any]] = []
        sft: list[dict[str, Any]] = []
        preferences: list[dict[str, Any]] = []
        online_rl: list[dict[str, Any]] = []
        for seed_record in splits[split]:
            teacher_log = None
            teacher_chunks = None
            for policy_name in POLICY_NAMES:
                policy_seed = _stable_integer(
                    seed,
                    seed_record.episode_id,
                    policy_name,
                    "behavior_policy",
                    modulo=2**31 - 1,
                )
                log, chunks = _rollout(
                    seed_record,
                    product_catalog,
                    policy_name=policy_name,
                    generation_seed=policy_seed,
                    prefer_external_policies=prefer_external_policies,
                    calibration=calibration_payload,
                )
                logs.append(log)
                policy_source_counts[log["policy_source"]] += 1
                if policy_name == "Teacher":
                    teacher_log, teacher_chunks = log, chunks
            assert teacher_log is not None and teacher_chunks is not None
            stage_rows = _sft_rows(seed_record, teacher_chunks)
            sft.extend(stage_rows)
            for row in stage_rows:
                curriculum_counts[split][row["curriculum_stage"]] += 1
            preferences.append(
                _preference_row(
                    seed_record,
                    product_catalog,
                    teacher_log,
                    calibration=calibration_payload,
                )
            )
            online_rl.append(
                _online_rl_row(
                    seed_record,
                    product_catalog,
                    calibration=calibration_payload,
                )
            )

        write_jsonl(
            target / ARTIFACT_PATHS["mixed_policy_logs"] / f"{split}.jsonl",
            logs,
        )
        write_jsonl(
            target / ARTIFACT_PATHS["sft_trajectories"] / f"{split}.jsonl",
            sft,
        )
        for stage in CURRICULUM_STAGES:
            write_jsonl(
                target
                / ARTIFACT_PATHS["sft_trajectories"]
                / f"{split}.{stage}.jsonl",
                (row for row in sft if row["curriculum_stage"] == stage),
            )
        write_jsonl(
            target / ARTIFACT_PATHS["preference_pairs"] / f"{split}.jsonl",
            preferences,
        )
        write_jsonl(
            target / ARTIFACT_PATHS["online_rl_tasks"] / f"{split}.jsonl",
            online_rl,
        )
        artifact_counts["mixed_policy_logs"][split] = len(logs)
        artifact_counts["sft_trajectories"][split] = len(sft)
        artifact_counts["preference_pairs"][split] = len(preferences)
        artifact_counts["online_rl_tasks"][split] = len(online_rl)

    config = {
        "dataset_version": DATASET_VERSION,
        "episodes": episodes,
        "feed_length": feed_length,
        "seed": seed,
        "split_counts": _split_counts(episodes),
        "behavior_policies": list(POLICY_NAMES),
        "prefer_external_policies": bool(prefer_external_policies),
        "behavior_calibration": calibration_payload,
        "artifact_counts": artifact_counts,
        "curriculum_stage_counts": curriculum_counts,
        "policy_source_counts": dict(sorted(policy_source_counts.items())),
    }
    manifest = build_manifest(
        output_dir=target,
        config=config,
        split_summary=split_summary,
        source_catalog=_catalog_source(product_catalog, source_path),
        include_paths=ARTIFACT_PATHS.values(),
    )
    manifest.pop("manifest_content_sha256")
    manifest["artifact_counts"] = artifact_counts
    manifest["curriculum_stage_counts"] = curriculum_counts
    manifest["manifest_content_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    write_json_atomic(target / "manifest.json", manifest)
    return manifest


__all__ = [
    "ARTIFACT_PATHS",
    "CURRICULUM_STAGES",
    "DATASET_VERSION",
    "FEED_SYSTEM_PROMPT",
    "MAX_FEED_LENGTH",
    "MIN_FEED_LENGTH",
    "POLICY_NAMES",
    "SFT_MESSAGE_CHAR_BUDGET",
    "SFT_SHORT_WINDOW_MAX_STEPS",
    "SPLIT_NAMES",
    "generate_episode_splits",
    "generate_feed_artifacts",
    "load_product_catalog",
]
