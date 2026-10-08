"""Component C: exact executable labels, conservative option resolution and recovery."""

import json
import re
from copy import deepcopy

from shopping_grpo.environment.actions import (
    NAVIGATION_BUTTONS,
    action_guard_tool_message,
    clickable_buttons,
)
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.purchase_support import field


def page_type(observation):
    match = re.search(r"^page_type: (\w+)", observation, re.MULTILINE)
    return match[1] if match else "unknown"


def option_map(observation):
    if page_type(observation) != "product_detail":
        return {}
    options = field(observation, "available_options", {})
    asin = re.search(r"^asin: (\S+)", observation, re.MULTILINE)
    if not isinstance(options, dict) or not asin:
        return {}
    buttons = set(clickable_buttons(observation))
    result = {}
    for axis, values in sorted(options.items()):
        if not isinstance(values, list):
            continue
        for value in values:
            if value in buttons and value.casefold() not in NAVIGATION_BUTTONS:
                token = f"opt:{asin[1]}:{len(result) + 1}"
                result[token] = {"axis": axis, "value": value}
    return result


def resolve_option_call(tool_call, observation):
    """Resolve only current-page IDs or a unique whitespace-only spelling."""
    call = deepcopy(tool_call)
    fn = call.get("function") or {}
    if fn.get("name") != "select_option":
        return call, None
    try:
        args = (
            json.loads(fn["arguments"])
            if isinstance(fn["arguments"], str)
            else dict(fn["arguments"])
        )
    except (KeyError, ValueError, TypeError):
        return call, None
    if set(args) != {"value"} or not isinstance(args["value"], str):
        return call, None
    requested = args["value"]
    mapping = option_map(observation)
    if requested in mapping:
        resolved = mapping[requested]["value"]
        method = "current_page_option_id"
    else:
        norm = lambda s: re.sub(r"\s+", "", s).casefold()
        choices = list(dict.fromkeys(v["value"] for v in mapping.values()))
        if requested in choices:
            return call, None
        candidates = [v for v in choices if norm(v) == norm(requested)]
        if len(candidates) != 1:
            return call, None
        resolved = candidates[0]
        method = "unique_whitespace_match"
    args["value"] = resolved
    fn["arguments"] = json.dumps(args, ensure_ascii=False)
    return call, {"method": method, "requested": requested, "resolved": resolved}


def recovery_message(observation, reason):
    state = page_type(observation)
    labels = {
        "search_results": "搜索结果页，需先open_product打开商品；此处不能查看商品子页或选择规格",
        "product_detail": "商品详情页；不存在Attributes按钮时跳过，不要为看Reviews先退到搜索页",
        "information_subpage": "信息子页，先prev_page返回一次",
        "search_home": "搜索首页，可以search_products",
    }
    return f"当前状态：{labels.get(state, state)}。被拒原因：{reason}。按本轮提供的合法工具改正，拒绝的动作没有改变页面。"


def render_exact_observation(state):
    # Replace only the action footer; do not expose automatic product details.
    text = render_structured_observation(state)
    actions = state.get("actions") or []
    if not isinstance(actions, list) or not all(isinstance(a, str) for a in actions):
        raise ValueError("actions must be a list of strings")
    prefix, separator, _ = text.rpartition("可点击的按钮: ")
    if not separator:
        raise ValueError("missing action footer")
    return prefix + separator + json.dumps(actions, ensure_ascii=False)


def interface_feedback(call, reason, observation):
    feedback = action_guard_tool_message(call, reason, observation)["content"]
    if reason == "search_not_available_on_current_page" and "Back to Search" in clickable_buttons(
        observation
    ):
        feedback = feedback.replace(
            "下一步只能从当前页面列出的目标中选择。",
            "当前页面不能搜索。先调用 back_to_search({})，看到搜索功能可用后再调用 search_products。",
        )
    return feedback + "\n" + recovery_message(observation, reason)
