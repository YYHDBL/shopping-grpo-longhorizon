"""把环境的结构化状态渲染成模型可见的稳定 observation。

渲染器只输出公开商品信息、当前页面和可执行按钮；它会主动拒绝 goal、reward
等隐藏字段，防止训练和评测把答案泄露给模型。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from shopping_grpo.environment.product_id import is_product_id

OBSERVATION_VERSION = "shopping-observation-v2"
HEADER = "[SHOPPING_OBSERVATION_V2]"


class StructuredObservationError(ValueError):
    """The environment supplied a malformed or unsafe public state."""


def _text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _list(value):
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _footer(state):
    # Action labels are executable identifiers, not prose. Preserve whitespace.
    actions = state.get("actions") or []
    if not isinstance(actions, list) or not all(isinstance(action, str) for action in actions):
        raise StructuredObservationError("actions must be a list of strings")
    return [
        f"搜索功能是否可用: {bool(state.get('search_available'))}",
        "可点击的按钮: " + json.dumps(actions, ensure_ascii=False),
    ]


def render_structured_observation(state: Mapping) -> str:
    """渲染一个状态，并拒绝版本不匹配或包含隐藏字段的输入。"""
    if not isinstance(state, Mapping):
        raise StructuredObservationError("observation_state must be an object")
    if state.get("observation_version") != OBSERVATION_VERSION:
        raise StructuredObservationError("unsupported observation_state version")
    forbidden = {"goal", "reward", "reward_detail", "target_asin", "answer"}
    leaked = forbidden.intersection(state)
    if leaked:
        raise StructuredObservationError(
            "observation_state contains forbidden fields: " + ", ".join(sorted(leaked))
        )

    # 先按页面类型渲染正文，最后统一追加搜索状态和按钮 footer；守卫和投影器
    # 都依赖这个 footer 来判断下一步动作是否合法。
    page_type = str(state.get("page_type") or "unknown")
    lines = [HEADER, f"page_type: {page_type}"]
    if page_type == "search_home":
        lines.append("使用 search_products 提交简短、具有区分度的商品查询。")
    elif page_type == "search_results":
        lines.extend(_render_search_results(state))
    elif page_type in {"product_detail", "information_subpage"}:
        lines.extend(_render_product(state))
        if page_type == "information_subpage":
            lines.append(f"subpage: {_text(state.get('subpage'))}")
            lines.append("content: " + _text(state.get("content")))
    elif page_type != "terminal":
        raise StructuredObservationError(f"unsupported page_type: {page_type!r}")
    return "\n".join(lines) + "\n\n" + "\n".join(_footer(state))


def _render_search_results(state):
    """渲染搜索结果，并确保每个展示的 ASIN 都是可操作目标。"""
    products = state.get("products")
    if not isinstance(products, list):
        raise StructuredObservationError("search results must contain a products list")
    actions = set(_list(state.get("actions")))
    product_asins = []
    lines = [
        f"query: {_text(state.get('query'))}",
        f"normalized_query: {_text(state.get('normalized_query'))}",
        (
            f"Page {int(state.get('page', 1))} of {int(state.get('total_pages', 1))} "
            f"(Total results: {int(state.get('total_results', 0))}; "
            f"ranks {int(state.get('rank_start', 0))}-{int(state.get('rank_end', 0))})"
        ),
        "格式: rank|asin|price|brand|category|key_attributes|title",
    ]
    for product in products:
        if not isinstance(product, Mapping):
            raise StructuredObservationError("each product must be an object")
        asin = _text(product.get("asin"))
        if not is_product_id(asin):
            raise StructuredObservationError(f"invalid search-result ASIN: {asin!r}")
        product_asins.append(asin)
        attributes = ",".join(_list(product.get("key_attributes")))
        lines.append(
            "|".join(
                (
                    str(int(product.get("rank", 0))),
                    asin,
                    _text(product.get("price")),
                    _text(product.get("brand")),
                    _text(product.get("category")),
                    attributes,
                    _text(product.get("title")),
                )
            )
        )
    actionable_asins = {action for action in actions if is_product_id(action)}
    if set(product_asins) != actionable_asins:
        raise StructuredObservationError(
            "model-visible search ASINs differ from environment-actionable ASINs"
        )
    if len(product_asins) > 20:
        raise StructuredObservationError("search page exceeds the frozen page size of 20")
    lines.append(f"products_shown: {len(product_asins)}")
    return lines


def _render_product(state):
    """渲染商品详情和规格选择，价格优先使用当前已选 variant 的价格。"""
    product = state.get("product")
    if not isinstance(product, Mapping):
        raise StructuredObservationError("product page must contain a product object")
    asin = _text(product.get("asin"))
    if not is_product_id(asin):
        raise StructuredObservationError(f"invalid product ASIN: {asin!r}")
    lines = [
        f"asin: {asin}",
        f"title: {_text(product.get('title'))}",
        f"brand: {_text(product.get('brand'))}",
        f"category: {_text(product.get('category'))}",
        f"price: {_text(state.get('selected_price', product.get('price')))}",
        "key_attributes: " + ", ".join(_list(product.get("key_attributes"))),
        "selected_options: "
        + json.dumps(state.get("selected_options") or {}, ensure_ascii=False, sort_keys=True),
        "available_options: "
        + json.dumps(state.get("available_options") or {}, ensure_ascii=False, sort_keys=True),
    ]
    public = state.get("public_details")
    if isinstance(public, Mapping):
        # Whitelist public fields; unknown/hidden payload keys are not rendered.
        for key in ("description", "features", "unselected_axes", "option_prices", "price_note", "option_evidence"):
            if key in public:
                lines.append("public_" + key + ": " + json.dumps(public[key], ensure_ascii=False))
    return lines
