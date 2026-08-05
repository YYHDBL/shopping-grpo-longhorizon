"""Deterministic product catalog and dependency-free hybrid retrieval."""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from shopping_grpo.feed.schema import Product, iter_jsonl, write_jsonl


_WORD_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
_CJK_RE = re.compile(r"^[\u3400-\u9fff]+$")


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _terms(text: str) -> Counter[str]:
    """Tokenize Latin words and CJK character n-grams without third-party NLP."""

    result: Counter[str] = Counter()
    for token in _WORD_RE.findall(_normalized(text)):
        if _CJK_RE.fullmatch(token):
            result.update(token)
            result.update(token[index : index + 2] for index in range(len(token) - 1))
            if len(token) <= 8:
                result[token] += 1
        else:
            result[token] += 1
    return result


def _coverage(query: Counter[str], document: Counter[str]) -> float:
    if not query or not document:
        return 0.0
    matched = sum(min(count, document.get(term, 0)) for term, count in query.items())
    return matched / sum(query.values())


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _coerce_embedding(value: object, *, product_id: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"embedding for {product_id!r} must be a sequence")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"embedding for {product_id!r} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"embedding for {product_id!r} contains a non-finite number")
        vector.append(number)
    if not vector:
        raise ValueError(f"embedding for {product_id!r} must not be empty")
    return tuple(vector)


def load_embeddings(
    source: Mapping[str, Sequence[float]] | str | Path,
) -> dict[str, tuple[float, ...]]:
    """Load ``product_id -> vector`` from a mapping, JSON object, or JSONL rows."""

    if isinstance(source, Mapping):
        return {
            str(product_id): _coerce_embedding(vector, product_id=str(product_id))
            for product_id, vector in source.items()
        }

    path = Path(source)
    suffixes = path.suffixes
    is_jsonl = ".jsonl" in suffixes
    if is_jsonl:
        result: dict[str, tuple[float, ...]] = {}
        for row in iter_jsonl(path):
            product_id = row.get("product_id", row.get("asin", row.get("id")))
            if not isinstance(product_id, str) or not product_id.strip():
                raise ValueError(f"embedding row in {path} is missing product_id")
            if product_id in result:
                raise ValueError(f"duplicate embedding for product {product_id!r}")
            result[product_id] = _coerce_embedding(row.get("embedding"), product_id=product_id)
        return result

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        return {
            str(product_id): _coerce_embedding(vector, product_id=str(product_id))
            for product_id, vector in payload.items()
        }
    if isinstance(payload, list):
        result = {}
        for row in payload:
            if not isinstance(row, Mapping):
                raise ValueError("embedding JSON array must contain objects")
            product_id = row.get("product_id", row.get("asin", row.get("id")))
            if not isinstance(product_id, str) or not product_id.strip():
                raise ValueError(f"embedding row in {path} is missing product_id")
            if product_id in result:
                raise ValueError(f"duplicate embedding for product {product_id!r}")
            result[product_id] = _coerce_embedding(row.get("embedding"), product_id=product_id)
        return result
    raise ValueError("embedding file must contain an object or an array of objects")


class ProductCatalog:
    """An immutable-view catalog with deterministic lexical/vector retrieval.

    Product order never affects results: records are indexed by ``product_id`` and all
    ties use that identifier.  Passing ``query_embedding`` activates hybrid retrieval
    for products with a precomputed embedding; no embedding model is imported or run.
    """

    def __init__(
        self,
        products: Iterable[Product | Mapping[str, object]]
        | Mapping[str, Product | Mapping[str, object]],
        embeddings: Mapping[str, Sequence[float]] | str | Path | None = None,
    ) -> None:
        if isinstance(products, Mapping):
            source: Iterable[tuple[str | None, Product | Mapping[str, object]]] = (
                (str(key), value) for key, value in products.items()
            )
        else:
            source = ((None, value) for value in products)

        by_id: dict[str, Product] = {}
        for mapping_id, raw in source:
            product = raw if isinstance(raw, Product) else Product.from_dict(raw)
            if mapping_id is not None and mapping_id != product.product_id:
                raise ValueError(
                    f"catalog key {mapping_id!r} does not match product_id {product.product_id!r}"
                )
            if product.product_id in by_id:
                raise ValueError(f"duplicate product_id {product.product_id!r}")
            by_id[product.product_id] = product

        supplied_embeddings = load_embeddings(embeddings) if embeddings is not None else {}
        unknown_embeddings = sorted(set(supplied_embeddings) - set(by_id))
        if unknown_embeddings:
            preview = ", ".join(unknown_embeddings[:3])
            raise ValueError(f"embeddings reference unknown products: {preview}")
        for product_id, vector in supplied_embeddings.items():
            by_id[product_id] = replace(by_id[product_id], embedding=vector)

        self._by_id = {product_id: by_id[product_id] for product_id in sorted(by_id)}
        self._products = tuple(self._by_id.values())
        self._search_fields = {
            product.product_id: (
                _normalized(product.title),
                _normalized(" ".join((product.category, product.domain))),
                _normalized(
                    " ".join(
                        (
                            *product.attributes,
                            *product.tags,
                            product.description,
                            product.brand,
                            product.shop_name,
                        )
                    )
                ),
            )
            for product in self._products
        }
        self._term_fields = {
            product_id: tuple(_terms(field) for field in fields)
            for product_id, fields in self._search_fields.items()
        }

    def __len__(self) -> int:
        return len(self._products)

    def __iter__(self) -> Iterator[Product]:
        return iter(self._products)

    def __contains__(self, product_id: object) -> bool:
        return product_id in self._by_id

    @property
    def products(self) -> tuple[Product, ...]:
        return self._products

    @property
    def product_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def get(self, product_id: str, default: Product | None = None) -> Product | None:
        return self._by_id.get(product_id, default)

    def require(self, product_id: str) -> Product:
        try:
            return self._by_id[product_id]
        except KeyError as exc:
            raise KeyError(f"unknown product_id {product_id!r}") from exc

    def _lexical_score(self, product: Product, query: str, query_terms: Counter[str]) -> float:
        title, category, details = self._search_fields[product.product_id]
        title_terms, category_terms, detail_terms = self._term_fields[product.product_id]
        score = (
            0.55 * _coverage(query_terms, title_terms)
            + 0.25 * _coverage(query_terms, category_terms)
            + 0.20 * _coverage(query_terms, detail_terms)
        )
        compact_query = query.replace(" ", "")
        if compact_query:
            if compact_query in title.replace(" ", ""):
                score += 0.30
            elif compact_query in category.replace(" ", ""):
                score += 0.15
            elif compact_query in details.replace(" ", ""):
                score += 0.08
        return min(score, 1.0)

    def search_with_scores(
        self,
        query: str = "",
        *,
        top_k: int = 10,
        query_embedding: Sequence[float] | None = None,
        lexical_weight: float = 0.65,
        embedding_weight: float = 0.35,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        product_ids: Iterable[str] | None = None,
    ) -> list[tuple[Product, float]]:
        """Return products and normalized hybrid scores in deterministic order."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if top_k == 0:
            return []
        normalized_query = _normalized(query)
        vector = (
            _coerce_embedding(query_embedding, product_id="<query>")
            if query_embedding is not None
            else None
        )
        if not normalized_query and vector is None:
            raise ValueError("search requires a non-empty query or query_embedding")
        if lexical_weight < 0.0 or embedding_weight < 0.0:
            raise ValueError("retrieval weights must be non-negative")
        if normalized_query and vector is not None and lexical_weight + embedding_weight == 0.0:
            raise ValueError("at least one retrieval weight must be positive")
        if normalized_query and vector is None and lexical_weight == 0.0:
            raise ValueError("lexical_weight must be positive for text-only search")
        if vector is not None and not normalized_query and embedding_weight == 0.0:
            raise ValueError("embedding_weight must be positive for vector-only search")
        if min_price is not None and min_price < 0.0:
            raise ValueError("min_price must be non-negative")
        if max_price is not None and max_price < 0.0:
            raise ValueError("max_price must be non-negative")
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ValueError("min_price must not exceed max_price")

        allowed_ids = set(product_ids) if product_ids is not None else None
        category_filter = _normalized(category or "")
        query_terms = _terms(normalized_query)
        scored: list[tuple[Product, float]] = []
        for product in self._products:
            if allowed_ids is not None and product.product_id not in allowed_ids:
                continue
            if category_filter and category_filter not in _normalized(product.category):
                continue
            if min_price is not None and product.price < min_price:
                continue
            if max_price is not None and product.price > max_price:
                continue

            lexical = (
                self._lexical_score(product, normalized_query, query_terms)
                if normalized_query
                else 0.0
            )
            cosine = (
                _cosine(vector, product.embedding)
                if vector is not None and product.embedding is not None
                else None
            )
            vector_score = max(0.0, min(1.0, (cosine + 1.0) / 2.0)) if cosine is not None else 0.0

            active_lexical_weight = lexical_weight if normalized_query else 0.0
            active_vector_weight = embedding_weight if cosine is not None else 0.0
            denominator = active_lexical_weight + active_vector_weight
            if denominator == 0.0:
                continue
            score = (
                active_lexical_weight * lexical + active_vector_weight * vector_score
            ) / denominator
            if score > 0.0:
                scored.append((product, score))

        scored.sort(
            key=lambda item: (
                -item[1],
                -item[0].popularity,
                item[0].price,
                item[0].product_id,
            )
        )
        return scored[:top_k]

    def search(
        self,
        query: str = "",
        *,
        limit: int | None = None,
        candidate_ids: Iterable[str] | None = None,
        video: object | None = None,
        persona: object | None = None,
        **kwargs: object,
    ) -> list[Product]:
        """Return products using both catalog and feed-simulator argument names.

        ``limit``/``candidate_ids`` are aliases for ``top_k``/``product_ids``.
        When a structured video carries a precomputed embedding it is used as the
        query vector unless the caller supplied ``query_embedding`` explicitly.
        ``persona`` is accepted for the simulator protocol but is deliberately not
        folded into catalog relevance; policy-level personalization remains auditable.
        """

        del persona
        if limit is not None:
            if "top_k" in kwargs:
                raise TypeError("pass only one of limit and top_k")
            kwargs["top_k"] = limit
        if candidate_ids is not None:
            if "product_ids" in kwargs:
                raise TypeError("pass only one of candidate_ids and product_ids")
            kwargs["product_ids"] = candidate_ids
        if video is not None and "query_embedding" not in kwargs:
            embedding = (
                video.get("embedding")
                if isinstance(video, Mapping)
                else getattr(video, "embedding", None)
            )
            if embedding is not None:
                kwargs["query_embedding"] = embedding
        return [product for product, _ in self.search_with_scores(query, **kwargs)]

    def alternatives(
        self,
        product_id: str,
        *,
        top_k: int = 5,
        limit: int | None = None,
        max_price: float | None = None,
        cheaper_only: bool = False,
        query_embedding: Sequence[float] | None = None,
    ) -> list[Product]:
        """Find same-category substitutes while excluding the source product."""

        if limit is not None:
            if top_k != 5:
                raise TypeError("pass only one of limit and top_k")
            top_k = limit
        target = self.require(product_id)
        ceiling = target.price if cheaper_only else max_price
        if max_price is not None:
            ceiling = min(target.price, max_price) if cheaper_only else max_price
        query = " ".join((target.title, target.category, *target.attributes))
        candidates = self.search_with_scores(
            query,
            top_k=len(self),
            query_embedding=query_embedding,
            category=target.category,
            max_price=ceiling,
        )
        return [product for product, _ in candidates if product.product_id != product_id][:top_k]

    def complements(
        self, product_id: str, *, top_k: int = 5, limit: int | None = None
    ) -> list[Product]:
        """Return explicit complements first, then related cross-category products."""

        if limit is not None:
            if top_k != 5:
                raise TypeError("pass only one of limit and top_k")
            top_k = limit
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        target = self.require(product_id)
        selected: list[Product] = []
        seen = {product_id}
        explicit_ids = (*target.complement_product_ids, *target.related_product_ids)
        for related_id in explicit_ids:
            related = self.get(related_id)
            if related is not None and related.product_id not in seen:
                selected.append(related)
                seen.add(related.product_id)
                if len(selected) == top_k:
                    return selected

        # A catalog without relationship labels cannot establish true complementarity.
        # The conservative fallback only considers cross-category products sharing
        # explicit attribute/tag evidence, and returns no arbitrary popular products.
        evidence = set(_terms(" ".join((*target.attributes, *target.tags))))
        if not evidence:
            return selected
        source_leaf = _normalized(target.category).split("›")[-1]
        inferred: list[tuple[float, Product]] = []
        for product in self._products:
            if product.product_id in seen:
                continue
            if _normalized(product.category).split("›")[-1] == source_leaf:
                continue
            candidate_terms = set(_terms(" ".join((*product.attributes, *product.tags))))
            overlap = len(evidence & candidate_terms)
            if overlap:
                inferred.append((overlap / len(evidence), product))
        inferred.sort(key=lambda item: (-item[0], -item[1].popularity, item[1].product_id))
        selected.extend(product for _, product in inferred[: max(0, top_k - len(selected))])
        return selected

    def to_jsonl(self, path: str | Path, *, include_embeddings: bool = True) -> None:
        rows = (product.to_dict(include_embedding=include_embeddings) for product in self._products)
        write_jsonl(path, rows)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        embeddings: Mapping[str, Sequence[float]] | str | Path | None = None,
    ) -> "ProductCatalog":
        return cls((Product.from_dict(row) for row in iter_jsonl(path)), embeddings=embeddings)

    @classmethod
    def from_shopsimulator(
        cls,
        path: str | Path,
        *,
        embeddings: Mapping[str, Sequence[float]] | str | Path | None = None,
        limit: int | None = None,
        expected_count: int | None = None,
    ) -> "ProductCatalog":
        """Adapt the ShopSimulator JSON/JSON.GZ product archive.

        Only product truth is retained.  ``instructions`` and ``user_persona`` are
        intentionally discarded so hidden goals cannot enter retrieval evidence.
        """

        source_path = Path(path)
        opener = gzip.open if source_path.suffix == ".gz" else open
        with opener(source_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            raw_products = list(payload.values())
        elif isinstance(payload, list):
            raw_products = payload
        else:
            raise ValueError("ShopSimulator product archive must be a JSON array or object")
        if expected_count is not None and len(raw_products) != expected_count:
            raise ValueError(
                f"ShopSimulator archive contains {len(raw_products)} products; "
                f"expected {expected_count}"
            )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("limit must be a non-negative integer")
            raw_products = raw_products[:limit]

        products: list[Product] = []
        for row_number, raw in enumerate(raw_products, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"ShopSimulator product row {row_number} must be an object")
            product = Product.from_dict(
                {
                    "product_id": raw.get("asin"),
                    "title": raw.get("title"),
                    "category": raw.get("category", ""),
                    "pricing": raw.get("pricing", ()),
                    # The frozen archive contains a small number of blank attribute
                    # strings.  They carry no product truth and are removed at the
                    # adapter boundary; the public Product contract remains strict.
                    "attributes": [
                        item.strip()
                        for item in raw.get("attribute", ())
                        if isinstance(item, str) and item.strip()
                    ],
                    "description": raw.get("full_description", ""),
                    "brand": raw.get("brand", ""),
                    "shop_name": raw.get("shop_name", ""),
                    "domain": raw.get("domain_zh", raw.get("domain_en_long", "")),
                    "images": raw.get("images", ()),
                    "source": "shopsimulator",
                    "metadata": {
                        "tag": raw.get("tag"),
                        "sub_title": raw.get("sub_title"),
                        "customization_options": raw.get("customization_options", {}),
                    },
                }
            )
            products.append(product)
        return cls(products, embeddings=embeddings)


__all__ = ["ProductCatalog", "load_embeddings"]
