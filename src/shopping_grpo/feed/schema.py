"""Typed, JSON-safe contracts for the long-horizon feed environment.

The feed profile is intentionally parallel to the frozen ShopSimulator profile.  The
classes in this module contain only observable/feed data; simulator-only latent state
belongs in the simulator implementation rather than in an :class:`EpisodeSeed`.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Type, TypeVar


MAX_IDENTIFIER_CHARS = 128
MAX_EVIDENCE_ID_CHARS = 512

class Decision(str, Enum):
    """The ``when`` part of a feed intervention."""

    RECOMMEND = "recommend"
    DELAY = "delay"
    NO_RECOMMEND = "no_recommend"


class Surface(str, Enum):
    """The user-visible surface used by an intervention."""

    NONE = "none"
    PRODUCT_CARD = "product_card"
    COUPON = "coupon"
    REVIEW_SUMMARY = "review_summary"
    PRICE_COMPARISON = "price_comparison"
    SIMILAR_PRODUCTS = "similar_products"
    BUNDLE = "bundle"
    CREATOR_VIDEO = "creator_video"


class Strategy(str, Enum):
    """The ``how`` part of a recommendation."""

    NONE = "none"
    DIRECT = "direct"
    REVIEW_SUMMARY = "review_summary"
    PRICE_COMPARISON = "price_comparison"
    DISCOUNT = "discount"
    CHEAPER_ALTERNATIVE = "cheaper_alternative"
    SIMILAR = "similar"
    COMPLEMENT = "complement"
    BUNDLE = "bundle"


class Relationship(str, Enum):
    """How selected products relate to the current content/primary item."""

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"
    COMPLEMENT = "complement"
    BUNDLE = "bundle"


class EventType(str, Enum):
    """Observable user feedback emitted by the feed simulator."""

    IMPRESSION = "impression"
    WATCH = "watch"
    SKIP = "skip"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    CLICK = "click"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PURCHASE = "purchase"
    RETURN = "return"
    SESSION_BREAK = "session_break"
    SESSION_END = "session_end"


_EVENT_ALIASES = {
    "exposure": EventType.IMPRESSION,
    "expose": EventType.IMPRESSION,
    "view": EventType.WATCH,
    "dwell": EventType.WATCH,
    "swipe": EventType.SKIP,
    "swipe_away": EventType.SKIP,
    "cart": EventType.ADD_TO_CART,
    "add_cart": EventType.ADD_TO_CART,
    "remove_cart": EventType.REMOVE_FROM_CART,
    "buy": EventType.PURCHASE,
    "order": EventType.PURCHASE,
    "refund": EventType.RETURN,
}


def _text(
    value: object,
    *,
    name: str,
    required: bool = False,
    maximum_chars: int | None = None,
) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{name} must not be empty")
    if maximum_chars is not None and len(value) > maximum_chars:
        raise ValueError(f"{name} must be at most {maximum_chars} characters")
    return value


def _optional_text(
    value: object, *, name: str, maximum_chars: int | None = None
) -> str | None:
    if value is None:
        return None
    text = _text(value, name=name, maximum_chars=maximum_chars)
    return text or None


def _number(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_number(
    value: object, *, name: str, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    return _number(value, name=name, minimum=minimum)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strings(
    value: object,
    *,
    name: str,
    maximum_items: int | None = None,
    maximum_chars: int | None = None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if maximum_items is not None and len(value) > maximum_items:
        raise ValueError(f"{name} must contain at most {maximum_items} items")
    result = tuple(
        _text(
            item,
            name=f"{name}[]",
            required=True,
            maximum_chars=maximum_chars,
        )
        for item in value
    )
    return result


def _embedding(value: object, *, name: str = "embedding") -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    result = tuple(_number(item, name=f"{name}[]") for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _enum(value: object, enum_type: Type[Enum], *, name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"invalid {name} {value!r}; expected one of {allowed}") from exc


def _event_enum(value: object) -> EventType:
    if isinstance(value, EventType):
        return value
    if isinstance(value, str) and value in _EVENT_ALIASES:
        return _EVENT_ALIASES[value]
    return _enum(value, EventType, name="event_type")


def _json_value(value: object, *, name: str = "value") -> Any:
    """Return a detached, deterministic JSON value or fail early."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain non-finite numbers")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value, name=name)
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict(), name=name)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings")
            result[key] = _json_value(value[key], name=f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, name=f"{name}[]") for item in value]
    raise TypeError(f"{name} contains a non-JSON value of type {type(value).__name__}")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return _json_value(value, name=name)


def _pick(data: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        if name in data:
            return data[name]
    return default


@dataclass(frozen=True, slots=True)
class Product:
    """A feed-safe product record independent of ShopSimulator goal data."""

    product_id: str
    title: str
    category: str = ""
    price: float = 0.0
    max_price: float | None = None
    attributes: tuple[str, ...] = ()
    description: str = ""
    brand: str = ""
    shop_name: str = ""
    domain: str = ""
    image_urls: tuple[str, ...] = ()
    rating: float | None = None
    inventory: int | None = None
    popularity: float = 0.0
    review_summary: str = ""
    tags: tuple[str, ...] = ()
    related_product_ids: tuple[str, ...] = ()
    complement_product_ids: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "product_id",
            _text(
                self.product_id,
                name="product_id",
                required=True,
                maximum_chars=MAX_IDENTIFIER_CHARS,
            ),
        )
        object.__setattr__(self, "title", _text(self.title, name="title", required=True))
        object.__setattr__(self, "category", _text(self.category, name="category"))
        object.__setattr__(self, "price", _number(self.price, name="price", minimum=0.0))
        object.__setattr__(
            self,
            "max_price",
            _optional_number(self.max_price, name="max_price", minimum=0.0),
        )
        if self.max_price is not None and self.max_price < self.price:
            raise ValueError("max_price must be greater than or equal to price")
        for name in ("attributes", "image_urls", "tags"):
            object.__setattr__(self, name, _strings(getattr(self, name), name=name))
        for name in ("related_product_ids", "complement_product_ids"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name=name,
                    maximum_chars=MAX_IDENTIFIER_CHARS,
                ),
            )
        for name in ("description", "brand", "shop_name", "domain", "review_summary", "source"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self, "rating", _optional_number(self.rating, name="rating", minimum=0.0)
        )
        if self.inventory is not None:
            object.__setattr__(self, "inventory", _integer(self.inventory, name="inventory"))
        object.__setattr__(
            self, "popularity", _number(self.popularity, name="popularity", minimum=0.0)
        )
        object.__setattr__(self, "embedding", _embedding(self.embedding))
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    @property
    def asin(self) -> str:
        """ShopSimulator-compatible identifier alias."""

        return self.product_id

    @property
    def min_price(self) -> float:
        return self.price

    def to_dict(self, *, include_embedding: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {
            "product_id": self.product_id,
            "title": self.title,
            "category": self.category,
            "price": self.price,
            "max_price": self.max_price,
            "attributes": list(self.attributes),
            "description": self.description,
            "brand": self.brand,
            "shop_name": self.shop_name,
            "domain": self.domain,
            "image_urls": list(self.image_urls),
            "rating": self.rating,
            "inventory": self.inventory,
            "popularity": self.popularity,
            "review_summary": self.review_summary,
            "tags": list(self.tags),
            "related_product_ids": list(self.related_product_ids),
            "complement_product_ids": list(self.complement_product_ids),
            "source": self.source,
            "metadata": _json_value(self.metadata, name="metadata"),
        }
        if include_embedding:
            row["embedding"] = list(self.embedding) if self.embedding is not None else None
        return row

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Product":
        if not isinstance(data, Mapping):
            raise TypeError("product must be an object")
        pricing = _pick(data, "pricing", default=None)
        price = _pick(data, "price", "min_price", default=None)
        maximum = _pick(data, "max_price", default=None)
        if price is None and isinstance(pricing, Sequence) and not isinstance(pricing, str):
            prices = [
                _number(item, name="pricing[]", minimum=0.0)
                for item in pricing
                if not isinstance(item, bool) and isinstance(item, (int, float))
            ]
            if prices:
                price, maximum = min(prices), max(prices)
        if price is None:
            price = 0.0
        return cls(
            product_id=_text(
                _pick(data, "product_id", "asin", "id"), name="product_id", required=True
            ),
            title=_text(_pick(data, "title", "name"), name="title", required=True),
            category=_text(_pick(data, "category", default=""), name="category"),
            price=_number(price, name="price", minimum=0.0),
            max_price=_optional_number(maximum, name="max_price", minimum=0.0),
            attributes=_strings(
                _pick(data, "attributes", "attribute", "key_attributes", default=()),
                name="attributes",
            ),
            description=_text(
                _pick(data, "description", "full_description", default=""), name="description"
            ),
            brand=_text(_pick(data, "brand", default=""), name="brand"),
            shop_name=_text(_pick(data, "shop_name", default=""), name="shop_name"),
            domain=_text(
                _pick(data, "domain", "domain_zh", "domain_en_long", default=""), name="domain"
            ),
            image_urls=_strings(_pick(data, "image_urls", "images", default=()), name="image_urls"),
            rating=_optional_number(
                _pick(data, "rating", default=None), name="rating", minimum=0.0
            ),
            inventory=(
                None
                if _pick(data, "inventory", "stock", default=None) is None
                else _integer(_pick(data, "inventory", "stock"), name="inventory")
            ),
            popularity=_number(
                _pick(data, "popularity", default=0.0), name="popularity", minimum=0.0
            ),
            review_summary=_text(
                _pick(data, "review_summary", default=""), name="review_summary"
            ),
            tags=_strings(_pick(data, "tags", default=()), name="tags"),
            related_product_ids=_strings(
                _pick(data, "related_product_ids", "related_products", default=()),
                name="related_product_ids",
            ),
            complement_product_ids=_strings(
                _pick(data, "complement_product_ids", "complements", default=()),
                name="complement_product_ids",
            ),
            embedding=_embedding(_pick(data, "embedding", default=None)),
            source=_text(_pick(data, "source", default=""), name="source"),
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class Video:
    """A structured video/feed item; raw video decoding is deliberately out of scope."""

    video_id: str
    caption: str = ""
    scene: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    style: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    asr: str = ""
    ocr: str = ""
    creator_type: str = ""
    related_product_ids: tuple[str, ...] = ()
    duration_seconds: float | None = None
    embedding: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "video_id",
            _text(
                self.video_id,
                name="video_id",
                required=True,
                maximum_chars=MAX_IDENTIFIER_CHARS,
            ),
        )
        for name in ("caption", "asr", "ocr", "creator_type"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in ("scene", "objects", "style", "topics", "related_product_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "duration_seconds",
            _optional_number(self.duration_seconds, name="duration_seconds", minimum=0.0),
        )
        object.__setattr__(self, "embedding", _embedding(self.embedding))
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    @property
    def scenes(self) -> tuple[str, ...]:
        return self.scene

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "caption": self.caption,
            "scene": list(self.scene),
            "objects": list(self.objects),
            "style": list(self.style),
            "topics": list(self.topics),
            "asr": self.asr,
            "ocr": self.ocr,
            "creator_type": self.creator_type,
            "related_product_ids": list(self.related_product_ids),
            "duration_seconds": self.duration_seconds,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "metadata": _json_value(self.metadata, name="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Video":
        if not isinstance(data, Mapping):
            raise TypeError("video must be an object")
        scene = _pick(data, "scene", "scenes", default=())
        if isinstance(scene, str):
            scene = (scene,)
        return cls(
            video_id=_text(_pick(data, "video_id", "id"), name="video_id", required=True),
            caption=_text(_pick(data, "caption", "title", default=""), name="caption"),
            scene=_strings(scene, name="scene"),
            objects=_strings(
                _pick(data, "objects", "detected_objects", default=()), name="objects"
            ),
            style=_strings(_pick(data, "style", "styles", default=()), name="style"),
            topics=_strings(_pick(data, "topics", "tags", default=()), name="topics"),
            asr=_text(_pick(data, "asr", default=""), name="asr"),
            ocr=_text(_pick(data, "ocr", default=""), name="ocr"),
            creator_type=_text(_pick(data, "creator_type", default=""), name="creator_type"),
            related_product_ids=_strings(
                _pick(data, "related_product_ids", "related_products", default=()),
                name="related_product_ids",
            ),
            duration_seconds=_optional_number(
                _pick(data, "duration_seconds", "duration", default=None),
                name="duration_seconds",
                minimum=0.0,
            ),
            embedding=_embedding(_pick(data, "embedding", "video_embedding", default=None)),
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class Persona:
    """Observable persona features used to initialize an episode."""

    persona_id: str
    country: str = ""
    budget: float | None = None
    category_interests: tuple[str, ...] = ()
    style_preferences: tuple[str, ...] = ()
    price_sensitivity: float = 0.5
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "persona_id",
            _text(
                self.persona_id,
                name="persona_id",
                required=True,
                maximum_chars=MAX_IDENTIFIER_CHARS,
            ),
        )
        object.__setattr__(self, "country", _text(self.country, name="country"))
        object.__setattr__(
            self, "budget", _optional_number(self.budget, name="budget", minimum=0.0)
        )
        for name in ("category_interests", "style_preferences", "tags"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name=name,
                    maximum_items=32,
                    maximum_chars=160,
                ),
            )
        sensitivity = _number(self.price_sensitivity, name="price_sensitivity", minimum=0.0)
        if sensitivity > 1.0:
            raise ValueError("price_sensitivity must be between 0 and 1")
        object.__setattr__(self, "price_sensitivity", sensitivity)
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    @property
    def user_id(self) -> str:
        return self.persona_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "country": self.country,
            "budget": self.budget,
            "category_interests": list(self.category_interests),
            "style_preferences": list(self.style_preferences),
            "price_sensitivity": self.price_sensitivity,
            "tags": list(self.tags),
            "metadata": _json_value(self.metadata, name="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Persona":
        if not isinstance(data, Mapping):
            raise TypeError("persona must be an object")
        return cls(
            persona_id=_text(
                _pick(data, "persona_id", "user_id", "id"), name="persona_id", required=True
            ),
            country=_text(_pick(data, "country", default=""), name="country"),
            budget=_optional_number(
                _pick(data, "budget", "budget_max", default=None), name="budget", minimum=0.0
            ),
            category_interests=_strings(
                _pick(data, "category_interests", "categories", default=()),
                name="category_interests",
            ),
            style_preferences=_strings(
                _pick(data, "style_preferences", "styles", default=()),
                name="style_preferences",
            ),
            price_sensitivity=_number(
                _pick(data, "price_sensitivity", default=0.5),
                name="price_sensitivity",
                minimum=0.0,
            ),
            tags=_strings(_pick(data, "tags", default=()), name="tags"),
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


def _inventory(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    pairs: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = value  # type: ignore[assignment]
    else:
        raise TypeError("inventory must be an object or a sequence of pairs")
    normalized: dict[str, int] = {}
    for raw_pair in pairs:
        if not isinstance(raw_pair, Sequence) or len(raw_pair) != 2:
            raise TypeError("inventory entries must be [product_id, quantity] pairs")
        product_id = _text(
            raw_pair[0],
            name="inventory product_id",
            required=True,
            maximum_chars=MAX_IDENTIFIER_CHARS,
        )
        quantity = _integer(raw_pair[1], name=f"inventory[{product_id}]")
        normalized[product_id] = quantity
    return tuple(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class EpisodeSeed:
    """Serializable, reproducible input for one fixed-feed episode."""

    episode_id: str
    persona: Persona
    videos: tuple[Video, ...]
    product_ids: tuple[str, ...]
    seed: int
    inventory: tuple[tuple[str, int], ...] = ()
    session_breaks: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_id",
            _text(
                self.episode_id,
                name="episode_id",
                required=True,
                maximum_chars=MAX_IDENTIFIER_CHARS,
            ),
        )
        if not isinstance(self.persona, Persona):
            raise TypeError("persona must be a Persona")
        videos = tuple(self.videos)
        if not all(isinstance(video, Video) for video in videos):
            raise TypeError("videos must contain Video records")
        if not videos:
            raise ValueError("videos must not be empty")
        object.__setattr__(self, "videos", videos)
        object.__setattr__(
            self,
            "product_ids",
            _strings(
                self.product_ids,
                name="product_ids",
                maximum_chars=MAX_IDENTIFIER_CHARS,
            ),
        )
        object.__setattr__(self, "seed", _integer(self.seed, name="seed"))
        object.__setattr__(self, "inventory", _inventory(self.inventory))
        breaks = tuple(
            _integer(item, name="session_breaks[]", minimum=1)
            for item in self.session_breaks
        )
        if tuple(sorted(set(breaks))) != breaks:
            raise ValueError("session_breaks must be strictly increasing")
        if breaks and breaks[-1] >= len(videos):
            raise ValueError("session_breaks must be smaller than the number of videos")
        object.__setattr__(self, "session_breaks", breaks)
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "persona": self.persona.to_dict(),
            "videos": [video.to_dict() for video in self.videos],
            "product_ids": list(self.product_ids),
            "seed": self.seed,
            "inventory": dict(self.inventory),
            "session_breaks": list(self.session_breaks),
            "metadata": _json_value(self.metadata, name="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EpisodeSeed":
        if not isinstance(data, Mapping):
            raise TypeError("episode seed must be an object")
        raw_persona = _pick(data, "persona", "user")
        raw_videos = _pick(data, "videos", "feed")
        if not isinstance(raw_persona, Mapping):
            raise TypeError("persona must be an object")
        if isinstance(raw_videos, (str, bytes)) or not isinstance(raw_videos, Sequence):
            raise TypeError("videos must be a sequence")
        return cls(
            episode_id=_text(
                _pick(data, "episode_id", "seed_id", "id"), name="episode_id", required=True
            ),
            persona=Persona.from_dict(raw_persona),
            videos=tuple(Video.from_dict(row) for row in raw_videos),
            product_ids=_strings(
                _pick(data, "product_ids", "product_pool", default=()), name="product_ids"
            ),
            seed=_integer(_pick(data, "seed", "random_seed", default=0), name="seed"),
            inventory=_inventory(_pick(data, "inventory", default={})),
            session_breaks=tuple(
                _integer(item, name="session_breaks[]", minimum=1)
                for item in _pick(data, "session_breaks", default=())
            ),
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class FeedAction:
    """Structured When--What--How action produced by the policy."""

    decision: Decision
    surface: Surface = Surface.NONE
    strategy: Strategy = Strategy.NONE
    relationship: Relationship = Relationship.PRIMARY
    product_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _enum(self.decision, Decision, name="decision"))
        object.__setattr__(self, "surface", _enum(self.surface, Surface, name="surface"))
        object.__setattr__(self, "strategy", _enum(self.strategy, Strategy, name="strategy"))
        object.__setattr__(
            self, "relationship", _enum(self.relationship, Relationship, name="relationship")
        )
        object.__setattr__(
            self,
            "product_ids",
            _strings(
                self.product_ids,
                name="product_ids",
                maximum_chars=MAX_IDENTIFIER_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(
                self.evidence_ids,
                name="evidence_ids",
                maximum_chars=MAX_EVIDENCE_ID_CHARS,
            ),
        )
        object.__setattr__(self, "explanation", _text(self.explanation, name="explanation"))

    @property
    def reason(self) -> str:
        """Backward-compatible read-only alias for :attr:`explanation`."""

        return self.explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "surface": self.surface.value,
            "strategy": self.strategy.value,
            "relationship": self.relationship.value,
            "product_ids": list(self.product_ids),
            "evidence_ids": list(self.evidence_ids),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeedAction":
        """Parse a strict action object.

        This validates the wire representation and enum values.  Environment-specific
        legality (visible products, evidence provenance, inventory, and action timing)
        is deliberately left to the action guard.
        """

        if not isinstance(data, Mapping):
            raise TypeError("feed action must be an object")
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
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"feed action has unknown fields: {', '.join(unknown)}")
        if "decision" not in data:
            raise ValueError("feed action is missing decision")
        if "explanation" in data and "reason" in data:
            raise ValueError("feed action must not contain both explanation and reason")
        return cls(
            decision=_enum(data["decision"], Decision, name="decision"),
            surface=_enum(data.get("surface", Surface.NONE.value), Surface, name="surface"),
            strategy=_enum(data.get("strategy", Strategy.NONE.value), Strategy, name="strategy"),
            relationship=_enum(
                data.get("relationship", Relationship.PRIMARY.value),
                Relationship,
                name="relationship",
            ),
            product_ids=_strings(data.get("product_ids", ()), name="product_ids"),
            evidence_ids=_strings(data.get("evidence_ids", ()), name="evidence_ids"),
            explanation=_text(
                data.get("explanation", data.get("reason", "")), name="explanation"
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """One internal information-tool call made before a committed action."""

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    evidence_ids: tuple[str, ...] = ()
    step: int = 0
    call_id: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, name="tool_name", required=True)
        )
        object.__setattr__(self, "arguments", _mapping(self.arguments, name="arguments"))
        object.__setattr__(self, "result", _json_value(self.result, name="result"))
        object.__setattr__(self, "evidence_ids", _strings(self.evidence_ids, name="evidence_ids"))
        object.__setattr__(self, "step", _integer(self.step, name="step"))
        object.__setattr__(self, "call_id", _text(self.call_id, name="call_id"))
        object.__setattr__(self, "error", _optional_text(self.error, name="error"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": _json_value(self.arguments, name="arguments"),
            "result": _json_value(self.result, name="result"),
            "evidence_ids": list(self.evidence_ids),
            "step": self.step,
            "call_id": self.call_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ToolRecord":
        if not isinstance(data, Mapping):
            raise TypeError("tool record must be an object")
        return cls(
            tool_name=_text(_pick(data, "tool_name", "name"), name="tool_name", required=True),
            arguments=_mapping(
                _pick(data, "arguments", "parameters", default={}), name="arguments"
            ),
            result=_pick(data, "result", "observation", default=None),
            evidence_ids=_strings(_pick(data, "evidence_ids", default=()), name="evidence_ids"),
            step=_integer(_pick(data, "step", default=0), name="step"),
            call_id=_text(_pick(data, "call_id", "id", default=""), name="call_id"),
            error=_optional_text(_pick(data, "error", default=None), name="error"),
        )


@dataclass(frozen=True, slots=True)
class UserEvent:
    """An observable behavior event, optionally representing an aggregate count."""

    event_type: EventType
    step: int = 0
    video_id: str | None = None
    product_id: str | None = None
    dwell_seconds: float = 0.0
    value: float = 0.0
    count: int = 1
    timestamp_ms: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", _event_enum(self.event_type))
        object.__setattr__(self, "step", _integer(self.step, name="step"))
        object.__setattr__(self, "video_id", _optional_text(self.video_id, name="video_id"))
        object.__setattr__(self, "product_id", _optional_text(self.product_id, name="product_id"))
        object.__setattr__(
            self, "dwell_seconds", _number(self.dwell_seconds, name="dwell_seconds", minimum=0.0)
        )
        object.__setattr__(self, "value", _number(self.value, name="value"))
        object.__setattr__(self, "count", _integer(self.count, name="count", minimum=1))
        if self.timestamp_ms is not None:
            object.__setattr__(
                self, "timestamp_ms", _integer(self.timestamp_ms, name="timestamp_ms")
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "step": self.step,
            "video_id": self.video_id,
            "product_id": self.product_id,
            "dwell_seconds": self.dwell_seconds,
            "value": self.value,
            "count": self.count,
            "timestamp_ms": self.timestamp_ms,
            "metadata": _json_value(self.metadata, name="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UserEvent":
        if not isinstance(data, Mapping):
            raise TypeError("user event must be an object")
        return cls(
            event_type=_event_enum(_pick(data, "event_type", "type", "event")),
            step=_integer(_pick(data, "step", default=0), name="step"),
            video_id=_optional_text(_pick(data, "video_id", default=None), name="video_id"),
            product_id=_optional_text(
                _pick(data, "product_id", "asin", default=None), name="product_id"
            ),
            dwell_seconds=_number(
                _pick(data, "dwell_seconds", "watch_seconds", "duration", default=0.0),
                name="dwell_seconds",
                minimum=0.0,
            ),
            value=_number(_pick(data, "value", "amount", default=0.0), name="value"),
            count=_integer(_pick(data, "count", default=1), name="count", minimum=1),
            timestamp_ms=(
                None
                if _pick(data, "timestamp_ms", "timestamp", default=None) is None
                else _integer(
                    _pick(data, "timestamp_ms", "timestamp"), name="timestamp_ms"
                )
            ),
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Signed reward components; costs and penalties should be non-positive."""

    qualified_purchase: float = 0.0
    satisfaction: float = 0.0
    engagement: float = 0.0
    revenue: float = 0.0
    interruption_penalty: float = 0.0
    return_penalty: float = 0.0
    irrelevance_penalty: float = 0.0
    coupon_cost: float = 0.0
    unsupported_claim_penalty: float = 0.0
    other: float = 0.0

    def __post_init__(self) -> None:
        for name in (
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
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name=name))

    @property
    def total(self) -> float:
        return math.fsum(
            (
                self.qualified_purchase,
                self.satisfaction,
                self.engagement,
                self.revenue,
                self.interruption_penalty,
                self.return_penalty,
                self.irrelevance_penalty,
                self.coupon_cost,
                self.unsupported_claim_penalty,
                self.other,
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "qualified_purchase": self.qualified_purchase,
            "satisfaction": self.satisfaction,
            "engagement": self.engagement,
            "revenue": self.revenue,
            "interruption_penalty": self.interruption_penalty,
            "return_penalty": self.return_penalty,
            "irrelevance_penalty": self.irrelevance_penalty,
            "coupon_cost": self.coupon_cost,
            "unsupported_claim_penalty": self.unsupported_claim_penalty,
            "other": self.other,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RewardBreakdown":
        if not isinstance(data, Mapping):
            raise TypeError("reward breakdown must be an object")
        result = cls(
            **{
                name: _number(data.get(name, 0.0), name=name)
                for name in (
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
                )
            }
        )
        if "total" in data:
            expected = _number(data["total"], name="total")
            if not math.isclose(result.total, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("reward total does not equal the sum of its components")
        return result


@dataclass(frozen=True, slots=True)
class FeedTransition:
    """One replayable feed decision and its observable consequences."""

    step: int
    video_id: str
    action: FeedAction
    events: tuple[UserEvent, ...] = ()
    reward: RewardBreakdown = field(default_factory=RewardBreakdown)
    tool_records: tuple[ToolRecord, ...] = ()
    observation: Mapping[str, Any] = field(default_factory=dict)
    done: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _integer(self.step, name="step"))
        object.__setattr__(self, "video_id", _text(self.video_id, name="video_id", required=True))
        if not isinstance(self.action, FeedAction):
            raise TypeError("action must be a FeedAction")
        events = tuple(self.events)
        if not all(isinstance(event, UserEvent) for event in events):
            raise TypeError("events must contain UserEvent records")
        object.__setattr__(self, "events", events)
        if not isinstance(self.reward, RewardBreakdown):
            raise TypeError("reward must be a RewardBreakdown")
        records = tuple(self.tool_records)
        if not all(isinstance(record, ToolRecord) for record in records):
            raise TypeError("tool_records must contain ToolRecord records")
        object.__setattr__(self, "tool_records", records)
        object.__setattr__(self, "observation", _mapping(self.observation, name="observation"))
        if not isinstance(self.done, bool):
            raise TypeError("done must be a boolean")
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    @property
    def user_events(self) -> tuple[UserEvent, ...]:
        return self.events

    @property
    def tools(self) -> tuple[ToolRecord, ...]:
        return self.tool_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "video_id": self.video_id,
            "action": self.action.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "reward": self.reward.to_dict(),
            "tool_records": [record.to_dict() for record in self.tool_records],
            "observation": _json_value(self.observation, name="observation"),
            "done": self.done,
            "metadata": _json_value(self.metadata, name="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeedTransition":
        if not isinstance(data, Mapping):
            raise TypeError("feed transition must be an object")
        action = _pick(data, "action")
        reward = _pick(data, "reward", default={})
        events = _pick(data, "events", "user_events", default=())
        records = _pick(data, "tool_records", "tools", default=())
        if not isinstance(action, Mapping):
            raise TypeError("action must be an object")
        if not isinstance(reward, Mapping):
            raise TypeError("reward must be an object")
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise TypeError("events must be a sequence")
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("tool_records must be a sequence")
        return cls(
            step=_integer(_pick(data, "step", default=0), name="step"),
            video_id=_text(_pick(data, "video_id"), name="video_id", required=True),
            action=FeedAction.from_dict(action),
            events=tuple(UserEvent.from_dict(row) for row in events),
            reward=RewardBreakdown.from_dict(reward),
            tool_records=tuple(ToolRecord.from_dict(row) for row in records),
            observation=_mapping(_pick(data, "observation", default={}), name="observation"),
            done=data.get("done", False),  # type: ignore[arg-type]
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """A complete serializable long-horizon rollout."""

    episode_id: str
    transitions: tuple[FeedTransition, ...] = ()
    total_reward: float | None = None
    done: bool = True
    termination_reason: str = "feed_exhausted"
    metrics: Mapping[str, Any] = field(default_factory=dict)
    final_state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", _text(self.episode_id, name="episode_id", required=True)
        )
        transitions = tuple(self.transitions)
        if not all(isinstance(item, FeedTransition) for item in transitions):
            raise TypeError("transitions must contain FeedTransition records")
        object.__setattr__(self, "transitions", transitions)
        computed = math.fsum(item.reward.total for item in transitions)
        reward = (
            computed
            if self.total_reward is None
            else _number(self.total_reward, name="total_reward")
        )
        object.__setattr__(self, "total_reward", reward)
        if not isinstance(self.done, bool):
            raise TypeError("done must be a boolean")
        object.__setattr__(
            self, "termination_reason", _text(self.termination_reason, name="termination_reason")
        )
        for name in ("metrics", "final_state", "metadata"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name=name))

    @property
    def events(self) -> tuple[UserEvent, ...]:
        return tuple(event for transition in self.transitions for event in transition.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "transitions": [transition.to_dict() for transition in self.transitions],
            "total_reward": self.total_reward,
            "done": self.done,
            "termination_reason": self.termination_reason,
            "metrics": _json_value(self.metrics, name="metrics"),
            "final_state": _json_value(self.final_state, name="final_state"),
            "metadata": _json_value(self.metadata, name="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EpisodeResult":
        if not isinstance(data, Mapping):
            raise TypeError("episode result must be an object")
        transitions = _pick(data, "transitions", default=())
        if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
            raise TypeError("transitions must be a sequence")
        done = data.get("done", True)
        if not isinstance(done, bool):
            raise TypeError("done must be a boolean")
        return cls(
            episode_id=_text(_pick(data, "episode_id", "id"), name="episode_id", required=True),
            transitions=tuple(FeedTransition.from_dict(row) for row in transitions),
            total_reward=_optional_number(
                _pick(data, "total_reward", default=None), name="total_reward"
            ),
            done=done,
            termination_reason=_text(
                _pick(data, "termination_reason", default="feed_exhausted"),
                name="termination_reason",
            ),
            metrics=_mapping(_pick(data, "metrics", default={}), name="metrics"),
            final_state=_mapping(_pick(data, "final_state", default={}), name="final_state"),
            metadata=_mapping(_pick(data, "metadata", default={}), name="metadata"),
        )


T = TypeVar("T")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from UTF-8 JSONL (optionally gzip-compressed)."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {source}:{line_number} must be an object")
            yield row


def load_jsonl(path: str | Path, record_type: Type[T] | None = None) -> list[Any] | list[T]:
    """Load JSONL dictionaries or instantiate a type exposing ``from_dict``."""

    rows = list(iter_jsonl(path))
    if record_type is None:
        return rows
    factory = getattr(record_type, "from_dict", None)
    if factory is None:
        raise TypeError("record_type must expose from_dict")
    return [factory(row) for row in rows]


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    """Write deterministic UTF-8 JSONL (optionally gzip-compressed)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            if not isinstance(payload, Mapping):
                raise TypeError("each JSONL row must be an object or expose to_dict")
            handle.write(
                json.dumps(
                    _json_value(payload, name="row"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")


read_jsonl = load_jsonl


__all__ = [
    "Decision",
    "EpisodeResult",
    "EpisodeSeed",
    "EventType",
    "FeedAction",
    "FeedTransition",
    "Persona",
    "Product",
    "Relationship",
    "RewardBreakdown",
    "Strategy",
    "Surface",
    "ToolRecord",
    "UserEvent",
    "Video",
    "iter_jsonl",
    "load_jsonl",
    "read_jsonl",
    "write_jsonl",
]
