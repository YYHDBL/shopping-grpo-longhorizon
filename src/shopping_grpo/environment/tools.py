"""ShopSimulator 对模型公开的唯一工具定义。

这里同时保存两件必须保持同步的内容：模型看到的 JSON Schema，以及把工具调用
转换成环境 ``search[...]``/``click[...]``/``finish[...]`` 动作的映射。
"""

CLICK_TOOL_ACTIONS = {
    "open_product": ("asin", None),
    "select_option": ("value", None),
    "view_description": (None, "Description"),
    "view_features": (None, "Features"),
    "view_reviews": (None, "Reviews"),
    "view_attributes": (None, "Attributes"),
    "next_page": (None, "Next >"),
    "prev_page": (None, "< Prev"),
    "back_to_search": (None, "Back to Search"),
    "buy_now": (None, "Buy Now"),
}


def tool_call_to_action(name, parameters):
    """把一个标准 tool call 转成 ShopSimulator 能执行的字符串动作。"""
    parameters = parameters or {}
    if name == "search_products":
        return f"search[{parameters['query']}]"
    if name == "finish_without_purchase":
        return f"finish[{parameters['reason']}]"
    key, fixed_value = CLICK_TOOL_ACTIONS[name]
    value = fixed_value if fixed_value is not None else parameters[key]
    return f"click[{value}]"


def _schema(name, description, properties=None, required=None):
    """生成统一格式的 OpenAI function schema，并禁止额外参数。

    无参数时省略 properties 键（additionalProperties=False 已表达无参数；
    空对象与 null 都无法写入 parquet 的 struct 列）。
    """
    parameters = {
        "type": "object",
        "required": required or [],
        "additionalProperties": False,
    }
    if properties:
        parameters["properties"] = properties
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_INTERACTION_TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "仅当最新 observation 显示“搜索功能是否可用: True”时搜索。query 应简洁，包含品类和最有区分度的品牌、型号、功能或规格；不得重复相同查询或机械复制整段需求。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "open_product",
        "打开最新 observation 当前搜索结果中列出的候选商品以核验价格、功能和规格；asin 必须原样取自该页面。",
        {"asin": {"type": "string"}},
        ["asin"],
    ),
    _schema(
        "select_option",
        "为当前候选补齐所有影响可购买 variant 的必要规格轴；value 必须来自最新 observation，同一规格轴只选一个值，不得选择导航按钮或历史页面中的值。选择后应按完整 variant 的实际价格重新检查预算。",
        {"value": {"type": "string"}},
        ["value"],
    ),
    _schema("view_description", "仅当当前页面显示 Description 按钮且需要补充核验商品时打开；无参数，必须传 {}。"),
    _schema("view_features", "仅当当前页面显示 Features 按钮且需要核验核心功能时打开；无参数，必须传 {}。"),
    _schema("view_reviews", "仅当当前页面显示 Reviews 按钮且需要补充判断时打开；无参数，必须传 {}。"),
    _schema("view_attributes", "仅当当前页面显示 Attributes 按钮且需要核验关键属性时打开；无参数，必须传 {}。"),
    _schema("next_page", "仅当当前页面显示 Next > 且当前页没有合适候选时翻到下一页以发现新商品；无参数，必须传 {}。"),
    _schema("prev_page", "仅当当前页面显示 < Prev 按钮时返回上一页；无参数，必须传 {}。"),
    _schema("back_to_search", "仅当当前页面显示 Back to Search 按钮时返回搜索页；无参数，必须传 {}。"),
    _schema(
        "buy_now",
        "不可撤销的终止动作。仅当最新 observation 显示 Buy Now、品类正确、完整 variant 的实际价格在预算内，且按品牌、型号与核心功能、规格属性比较后是已核验候选中的最佳选择时购买；无参数，必须传 {}。",
    ),
]

_FINISH_WITHOUT_PURCHASE_SCHEMA = _schema(
    "finish_without_purchase",
    "主动结束且不购买，这不是成功。经过多次有实质差异的搜索和多个候选核验，仍没有可接受商品，并且当前没有明显值得继续核验的候选时调用；是否达到资格由环境判断。",
    {
        "reason": {
            "type": "string",
            "enum": ["no_suitable_product"],
        }
    },
    ["reason"],
)
SHOP_TOOL_SCHEMAS = [
    *_INTERACTION_TOOL_SCHEMAS,
    _FINISH_WITHOUT_PURCHASE_SCHEMA,
]
