"""在不丢失动作目标的前提下压缩模型可见 observation。

长商品标题和详情会占用上下文，但搜索结果中的 ASIN、页面导航和 footer 不能被
截掉。投影器因此先按页面类型选择预算，再压缩正文，最后验证所有可操作目标仍
与原始 observation 一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re

from shopping_grpo.environment.actions import (
    NAVIGATION_BUTTONS,
    clickable_buttons,
    product_ids,
)
from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE, is_product_id


FOOTER_MARKER = "\n\n搜索功能是否可用:"
TRUNCATION_MARKER = "[TRUNCATED_BY_SHOPPING_PROJECTOR]"
PROJECTION_CONTRACT_VERSION = "shopping-observation-v2"


class ObservationProjectionError(RuntimeError):
    """A safe model-visible observation cannot be produced."""


@dataclass(frozen=True)
class ObservationProjectionMeta:
    tool_name: str
    page_type: str
    token_budget: int
    raw_tokens: int
    visible_tokens: int
    truncation_ratio: float
    truncated: bool
    raw_asin_count: int
    visible_asin_count: int
    raw_button_count: int
    visible_button_count: int
    critical_footer_preserved: bool

    def to_dict(self):
        return asdict(self)


def project_observation(
    tool_name,
    observation,
    *,
    count_tokens,
    token_budget=3072,
    detail_token_budget=6144,
    generic_token_budget=1024,
    parameters=None,
    search_top_k=20,
):
    """返回有 token 上限的 observation，并保证可见目标与动作守卫目标一致。"""
    observation = str(observation)
    token_budget = int(token_budget)
    detail_token_budget = int(detail_token_budget)
    generic_token_budget = int(generic_token_budget)
    search_top_k = int(search_top_k)
    if min(token_budget, detail_token_budget, generic_token_budget) < 64:
        raise ValueError("all observation token budgets must be at least 64")
    if search_top_k < 1:
        raise ValueError("search_top_k must be positive")

    raw_tokens = int(count_tokens(observation))
    raw_buttons = clickable_buttons(observation)
    raw_asins = product_ids(observation)
    page_type = _page_type(observation)
    effective_budget = {
        "search_results": token_budget,
        "product_detail": detail_token_budget,
    }.get(page_type, generic_token_budget)
    # 短 observation 原样保留；只有超预算时才压缩，避免不必要地改变模型输入。
    if raw_tokens <= effective_budget:
        visible = observation
    else:
        _require_action_footer(observation)
        if page_type == "search_results":
            visible = _project_search_results(
                observation,
                parameters=parameters or {},
                count_tokens=count_tokens,
                token_budget=effective_budget,
                search_top_k=search_top_k,
            )
        else:
            visible = _project_generic_page(
                observation,
                count_tokens=count_tokens,
                token_budget=effective_budget,
            )

    # 压缩完成后再次验证：预算、footer、ASIN 和按钮都必须满足安全契约。
    visible_tokens = int(count_tokens(visible))
    if visible_tokens > effective_budget:
        raise ObservationProjectionError(
            f"visible observation uses {visible_tokens} tokens, above budget {effective_budget}"
        )
    visible_buttons = clickable_buttons(visible)
    visible_asins = product_ids(visible)
    footer_preserved = _critical_footer_preserved(
        observation,
        visible,
        raw_buttons=raw_buttons,
        visible_buttons=visible_buttons,
    )
    if not footer_preserved:
        raise ObservationProjectionError("critical search/button footer was not preserved")
    if page_type != "search_results" and set(visible_buttons) != set(raw_buttons):
        raise ObservationProjectionError("non-search projection changed actionable buttons")
    if page_type == "search_results":
        if set(raw_asins) != set(visible_asins):
            raise ObservationProjectionError(
                "search projection must preserve every product on the current environment page"
            )
        visible_product_targets = {
            button for button in visible_buttons if is_product_id(button)
        }
        if set(raw_asins) != visible_product_targets:
            raise ObservationProjectionError(
                "all current-page search products must remain visible and actionable"
            )

    meta = ObservationProjectionMeta(
        tool_name=str(tool_name),
        page_type=page_type,
        token_budget=effective_budget,
        raw_tokens=raw_tokens,
        visible_tokens=visible_tokens,
        truncation_ratio=(visible_tokens / raw_tokens if raw_tokens else 1.0),
        truncated=visible != observation,
        raw_asin_count=len(raw_asins),
        visible_asin_count=len(visible_asins),
        raw_button_count=len(raw_buttons),
        visible_button_count=len(visible_buttons),
        critical_footer_preserved=footer_preserved,
    )
    return visible, meta


def _page_type(observation):
    if (
        "[SHOPPING_OBSERVATION_V2]" in observation
        and "page_type: search_results" in observation
    ):
        return "search_results"
    if re.search(r"(?:^|\[SEP\]\s*)Page \d+", observation) and product_ids(observation):
        return "search_results"
    buttons = {button.casefold() for button in clickable_buttons(observation)}
    if "buy now" in buttons or "价格:" in observation:
        return "product_detail"
    if "搜索功能是否可用: True" in observation:
        return "search_home"
    if buttons and buttons <= {"back to search", "< prev"}:
        return "information_subpage"
    return "generic"


def _project_search_results(
    observation,
    *,
    parameters,
    count_tokens,
    token_budget,
    search_top_k,
):
    """压缩搜索结果标题，但保留当前页的每一个候选 ASIN。"""
    if "[SHOPPING_OBSERVATION_V2]" in observation:
        return _project_structured_search_results(
            observation,
            count_tokens=count_tokens,
            token_budget=token_budget,
            search_top_k=search_top_k,
        )
    body, footer = _split_footer(observation)
    segments = [segment.strip() for segment in body.split("[SEP]")]
    page = next((segment for segment in segments if re.fullmatch(r"Page \d+.*", segment)), "Page unknown")
    products = []
    for index, segment in enumerate(segments):
        if not is_product_id(segment):
            continue
        title = segments[index + 1] if index + 1 < len(segments) else ""
        price = segments[index + 2] if index + 2 < len(segments) else ""
        products.append((segment, title, price))
    if not products:
        return _project_generic_page(
            observation,
            count_tokens=count_tokens,
            token_budget=token_budget,
        )
    if len(products) > search_top_k:
        raise ObservationProjectionError(
            f"raw search page has {len(products)} products, above configured "
            f"page capacity {search_top_k}"
        )

    query = str(parameters.get("query", "")).strip()
    raw_buttons = clickable_buttons(observation)
    selected_asins = {asin for asin, _, _ in products}

    def render(title_character_limit):
        compacted_titles = [
            _compact_title(title, title_character_limit) for _, title, _ in products
        ]
        titles_compacted = any(
            compacted != title
            for compacted, (_, title, _) in zip(compacted_titles, products, strict=True)
        )
        visible_buttons = [
            button for button in raw_buttons
            if button.casefold() in NAVIGATION_BUTTONS or button in selected_asins
        ]
        lines = [
            "[SHOPPING_OBSERVATION_PROJECTION]",
            "page_type: search_results",
            f"query: {query or '(not recorded)'}",
            f"page: {page}",
            f"products_shown: {len(products)}/{len(products)}",
        ]
        lines.extend(
            f"{position:02d}|{asin}|{price}|{title}"
            for position, ((asin, _, price), title) in enumerate(
                zip(products, compacted_titles, strict=True),
                start=1,
            )
        )
        if titles_compacted:
            lines.append(f"{TRUNCATION_MARKER} product_titles_compacted=true")
        lines.extend(_footer_lines(footer, visible_buttons))
        return "\n".join(lines)

    maximum_title_characters = max((len(title) for _, title, _ in products), default=0)
    # 用二分搜索找到预算内最长标题，尽量保留信息而不引入复杂的排序策略。
    low, high, best = 0, maximum_title_characters, None
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if int(count_tokens(candidate)) <= token_budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is not None:
        return best
    raise ObservationProjectionError(
        "all current-page search products cannot fit the observation token budget, "
        "even with empty title snippets"
    )


def _project_structured_search_results(
    observation,
    *,
    count_tokens,
    token_budget,
    search_top_k,
):
    body, footer = _split_footer(observation)
    lines = body.splitlines()
    product_lines = []
    header_lines = []
    for line in lines:
        match = re.fullmatch(rf"(\d+)\|({PRODUCT_ID_CAPTURE})\|(.*)", line)
        if match:
            product_lines.append((match.group(1), match.group(2), match.group(3)))
        elif line and not line.startswith("products_shown:"):
            header_lines.append(line)
    if not product_lines:
        raise ObservationProjectionError(
            "structured search observation has no product rows"
        )
    if len(product_lines) > search_top_k:
        raise ObservationProjectionError(
            f"raw search page has {len(product_lines)} products, above configured "
            f"page capacity {search_top_k}"
        )
    raw_buttons = clickable_buttons(observation)
    product_asins = {asin for _, asin, _ in product_lines}
    visible_buttons = [
        button
        for button in raw_buttons
        if button in product_asins or button.casefold() in NAVIGATION_BUTTONS
    ]

    def render(character_limit):
        compacted = []
        truncated = False
        for rank, asin, payload in product_lines:
            fields = payload.split("|")
            compacted_fields = [
                _compact_title(field, character_limit) for field in fields
            ]
            truncated = truncated or compacted_fields != fields
            compacted.append("|".join((rank, asin, *compacted_fields)))
        rendered = [*header_lines, *compacted]
        if truncated:
            rendered.append(f"{TRUNCATION_MARKER} product_fields_compacted=true")
        rendered.extend(_footer_lines(footer, visible_buttons))
        return "\n".join(rendered)

    maximum = max(
        (len(field) for _, _, payload in product_lines for field in payload.split("|")),
        default=0,
    )
    low, high, best = 0, maximum, None
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if int(count_tokens(candidate)) <= token_budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ObservationProjectionError(
            "all structured current-page products cannot fit the observation token budget"
        )
    return best


def _compact_title(title, character_limit):
    title = str(title)
    character_limit = int(character_limit)
    if character_limit <= 0:
        return ""
    if len(title) <= character_limit:
        return title
    if character_limit == 1:
        return "…"
    if character_limit == 2:
        return title[0] + "…"
    head_length = max(1, (character_limit - 1) * 2 // 3)
    tail_length = max(1, character_limit - 1 - head_length)
    return title[:head_length] + "…" + title[-tail_length:]


def _project_generic_page(observation, *, count_tokens, token_budget):
    """压缩详情/子页正文，同时完整保留动作 footer。"""
    body, footer = _split_footer(observation)
    footer_lines = _footer_lines(footer, clickable_buttons(observation))
    suffix = "\n".join([TRUNCATION_MARKER, *footer_lines])
    if int(count_tokens(suffix)) > token_budget:
        raise ObservationProjectionError("critical footer exceeds the observation token budget")

    # Keep both ends of the page body: titles/key attributes are usually near the
    # front, while selected options and navigation context may be appended near
    # the end. The complete actionable footer is always preserved separately.
    low, high, best = 0, len(body), ""
    while low <= high:
        middle = (low + high) // 2
        tail_length = min(middle // 5, len(body))
        head_length = middle - tail_length
        if tail_length:
            projected_body = (
                body[:head_length].rstrip()
                + "\n"
                + TRUNCATION_MARKER
                + "\n"
                + body[-tail_length:].lstrip()
            )
            candidate = projected_body + "\n" + "\n".join(footer_lines)
        else:
            candidate = suffix
        if int(count_tokens(candidate)) <= token_budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise ObservationProjectionError("unable to fit generic observation into token budget")
    return best


def _split_footer(observation):
    index = observation.rfind(FOOTER_MARKER)
    if index < 0:
        return observation, ""
    return observation[:index], observation[index + 2 :]


def _require_action_footer(observation):
    if FOOTER_MARKER not in observation or "可点击的按钮:" not in observation:
        raise ObservationProjectionError(
            "long observation has no complete action footer and cannot be projected safely"
        )


def _footer_lines(footer, buttons):
    search_state = next(
        (
            line.strip()
            for line in footer.splitlines()
            if line.strip().startswith("搜索功能是否可用:")
        ),
        "搜索功能是否可用: False",
    )
    return [
        search_state,
        "可点击的按钮: " + json.dumps(buttons, ensure_ascii=False),
    ]


def _critical_footer_preserved(raw, visible, *, raw_buttons, visible_buttons):
    if "搜索功能是否可用:" in raw and "搜索功能是否可用:" not in visible:
        return False
    if "可点击的按钮:" in raw and "可点击的按钮:" not in visible:
        return False
    raw_navigation = {
        button.casefold() for button in raw_buttons if button.casefold() in NAVIGATION_BUTTONS
    }
    visible_navigation = {
        button.casefold() for button in visible_buttons if button.casefold() in NAVIGATION_BUTTONS
    }
    return raw_navigation == visible_navigation
