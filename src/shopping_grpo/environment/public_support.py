"""Opt-in, goal-free interaction support; never reads target or reward fields."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal

from shopping_grpo.environment.actions import NAVIGATION_BUTTONS, clickable_buttons, product_ids
from shopping_grpo.environment.budget import check_purchase_budget
from shopping_grpo.environment.tools import CLICK_TOOL_ACTIONS

VERSION = "public-interaction-support-v1"
RULES = """
逐项核验与执行补充：
- 原需求中的尺寸、数量、型号、颜色、配件分别核对最终selected_options；同一轴后选会替换前选，不会合并。标题描述整类商品，不能覆盖具体规格中的“不休眠”“不含”、单配件等差异。
- 数量与单位先换算再比较：1升=1000毫升，1公斤=2斤；克与毫升不能在没有密度时互换。“小于”不含等号。需要整机或礼盒时，单独配件不能替代。
- 每个需求只可标记“有页面证据”“明确冲突”“未知”。未知不是满足；代码型颜色没有释义不能猜颜色；破损赔偿不等于包装结实，温度传感器不自动等于高精度。不要把评分答案当作可见信息。
- 不把“左右”“约”变成固定上下限。价格明显偏离约数时继续比较；明确上下限必须检查。定金、尾款、单价、总价分开，规格文字和价格冲突时不得声称已核验。
- 没有Attributes按钮就跳过Attributes。在商品详情页想看Reviews时直接调用view_reviews，不能先prev_page退到搜索页。信息子页只返回一次；搜索结果页先open_product再选规格或看详情。
- 每个可选规格轴都显式完成，包括单值项；不要声称选过实际没有执行的值。核验不改变选择；如为比较价格切换过规格，购买前重新确认最终组合。
- Features为空或Description只是标题时，不再往返重复查看；当前有明显缺项就换有希望的候选或有实质差异的查询。多次有效探索仍不可核验则合理结束，不为凑成功强行购买。
- 工具被拒绝时先读当前页面与合法工具，不重复被拒调用。最终文本不是完成任务；环境结束前继续使用合法工具。
"""


def field(observation, name, default=None):
    match = re.search(r"^" + re.escape(name) + r":\s*([^\n]*)$", observation, re.MULTILINE)
    if not match:
        return default
    try:
        return json.loads(match[1])
    except json.JSONDecodeError:
        return default


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


def available_schemas(schemas, observation, *, option_labels=False):
    """Only constrain available tools, never choose a purchase or an option."""
    buttons = {b.casefold() for b in clickable_buttons(observation)}
    options = option_map(observation)
    result = []
    for original in schemas:
        schema = deepcopy(original)
        fn = schema["function"]
        name = fn["name"]
        if name == "think":
            continue
        if name == "search_products":
            allowed = "搜索功能是否可用: True" in observation
        elif name == "open_product":
            allowed = bool(product_ids(observation))
            fn["parameters"]["properties"]["asin"]["enum"] = product_ids(observation)
        elif name == "select_option":
            allowed = bool(options)
            fn["parameters"]["properties"]["value"]["enum"] = (
                list(dict.fromkeys(v["value"] for v in options.values()))
                if option_labels
                else list(options)
            )
            if not option_labels:
                fn["description"] += " 当前页面可用编号（只对此商品此页面有效）：" + json.dumps(
                    options, ensure_ascii=False, separators=(",", ":")
                )
        elif name == "buy_now":
            axes = field(observation, "available_options", {})
            selected = field(observation, "selected_options", {})
            complete = (
                isinstance(axes, dict)
                and isinstance(selected, dict)
                and all(not values or axis in selected for axis, values in axes.items())
            )
            allowed = "buy now" in buttons and complete
        elif name == "finish_without_purchase":
            allowed = True
        else:
            action = CLICK_TOOL_ACTIONS.get(name)
            allowed = bool(action and action[1] and action[1].casefold() in buttons)
        if allowed:
            result.append(schema)
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


def purchase_check(instruction, observation):
    available = field(observation, "available_options", {})
    selected = field(observation, "selected_options", {})
    if not isinstance(available, dict) or not isinstance(selected, dict):
        return {
            "reason": "variant_state_unreadable",
            "feedback": "无法读取当前规格状态，购买未执行。请重新打开商品核验。",
        }
    missing = {
        axis: values for axis, values in available.items() if values and axis not in selected
    }
    if missing:
        return {
            "reason": "variant_axes_unselected",
            "feedback": "购买未执行：这些公开规格轴尚未选择（单值也需显式选择）："
            + json.dumps(missing, ensure_ascii=False)
            + "。请在当前商品页补齐再核对价格。",
        }
    if not re.search(r"^price:[ \t]*\d+(?:\.\d+)?[ \t]*$", observation, re.MULTILINE):
        return {
            "reason": "variant_price_unknown",
            "feedback": "购买未执行：当前没有完整规格的单一确定价格，请核验规格和价格。",
        }
    result = check_purchase_budget(instruction, observation)
    if result["reason"]:
        return result
    # Cover additional explicit wording, never turn approximations into hard caps.
    bounds = []
    for clause in re.split(r"[，,。；;！？?\n]", instruction):
        if re.search(r"左右|上下|大概|大约|差不多|最好|约", clause):
            continue
        for pattern in [
            r"不超\s*(\d+(?:\.\d+)?)\s*(?:元|块)",
            r"(?:预算|价格)\s*(?:控制在|有|为|是)\s*(\d+(?:\.\d+)?)\s*(?:元|块)",
            r"(\d+(?:\.\d+)?)\s*(?:元|块)?能搞定",
        ]:
            bounds += [Decimal(m[1]) for m in re.finditer(pattern, clause)]
    price = Decimal(re.search(r"^price:[ \t]*(\d+(?:\.\d+)?)", observation, re.MULTILINE)[1])
    if bounds and price > min(bounds):
        return {
            "reason": "budget_above_explicit_amount",
            "feedback": f"购买未执行：当前{price}元超过公开明确预算{min(bounds)}元，请换规格或候选，不要重复购买。",
        }
    return {"reason": None, "feedback": None}


def recovery_message(observation, reason):
    state = page_type(observation)
    labels = {
        "search_results": "搜索结果页，需先open_product打开商品；此处不能查看商品子页或选择规格",
        "product_detail": "商品详情页；不存在Attributes按钮时跳过，不要为看Reviews先退到搜索页",
        "information_subpage": "信息子页，先prev_page返回一次",
        "search_home": "搜索首页，可以search_products",
    }
    return f"当前状态：{labels.get(state, state)}。被拒原因：{reason}。按本轮提供的合法工具改正，拒绝的动作没有改变页面。"
