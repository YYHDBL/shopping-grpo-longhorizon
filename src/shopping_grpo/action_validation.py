"""校验工具调用是否符合 ShopSimulator 当前 observation。"""

import json
import re

from shopping_grpo.shop_tools import tool_call_to_action


RUNTIME_GUARD_FIELD = "runtime_action_guard"


def action_reject_reason(name, arguments, observation):
    """返回动作拒绝原因；None 表示允许执行。"""
    if name == "think":
        return None
    if name == "search_products":
        if "搜索功能是否可用: False" in observation:
            return "search_not_available_on_current_page"
        return None
    if not observation:
        return "missing_previous_observation"
    if name == "open_product":
        asin = str(arguments.get("asin", ""))
        if asin not in product_ids(observation):
            return "click_not_in_previous_observation"
        return None

    try:
        action = tool_call_to_action(name, arguments)
    except Exception:
        return "unknown_or_invalid_tool"
    if not isinstance(action, str) or not action.startswith("click[") or not action.endswith("]"):
        return "click_not_in_previous_observation"
    target = action[6:-1].casefold()
    if target not in {button.casefold() for button in clickable_buttons(observation)}:
        return "click_not_in_previous_observation"
    return None


def action_guard_tool_message(tool_call, reason, observation):
    """返回标准 tool error observation，让 Teacher 感知刚才调用未被执行。"""
    targets = clickable_buttons(observation)
    asins = product_ids(observation)
    parts = []
    if asins:
        parts.append("可打开的商品 ASIN: " + ", ".join(asins[:20]))
    if targets:
        parts.append("可点击按钮: " + ", ".join(targets[:30]))
    allowed = "；".join(parts) or "当前页面没有可用点击目标"
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": (tool_call.get("function") or {}).get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": (
            f"上一工具调用被本地动作守卫拒绝（{reason}），未执行。"
            f"仅可依据当前页面重试。{allowed}。只调用一个合法工具。"
        ),
    }


def product_ids(observation):
    return list(dict.fromkeys(re.findall(r"(?<!\d)\d{12}(?!\d)", observation)))


def clickable_buttons(observation):
    match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", observation)
    if not match:
        return []
    try:
        buttons = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [button for button in buttons if isinstance(button, str)]
