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
    parameters = parameters or {}
    if name == "think":
        return None
    if name == "search_products":
        return f"search[{parameters['query']}]"
    key, fixed_value = CLICK_TOOL_ACTIONS[name]
    value = fixed_value if fixed_value is not None else parameters[key]
    return f"click[{value}]"


def _schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


SHOP_TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "仅当最新 observation 显示“搜索功能是否可用: True”时搜索；query 应包含品类和关键约束。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "open_product",
        "打开最新 observation 当前搜索结果中列出的商品；asin 必须原样取自该页面。",
        {"asin": {"type": "string"}},
        ["asin"],
    ),
    _schema(
        "select_option",
        "选择最新 observation 当前商品页列出的规格值；不得选择导航按钮或历史页面中的值。",
        {"value": {"type": "string"}},
        ["value"],
    ),
    _schema("view_description", "仅当当前页面显示 Description 按钮时打开；无参数，必须传 {}。"),
    _schema("view_features", "仅当当前页面显示 Features 按钮时打开；无参数，必须传 {}。"),
    _schema("view_reviews", "仅当当前页面显示 Reviews 按钮时打开；无参数，必须传 {}。"),
    _schema("view_attributes", "仅当当前页面显示 Attributes 按钮时打开；无参数，必须传 {}。"),
    _schema("next_page", "仅当当前页面显示 Next > 按钮时翻到下一页；无参数，必须传 {}。"),
    _schema("prev_page", "仅当当前页面显示 < Prev 按钮时返回上一页；无参数，必须传 {}。"),
    _schema("back_to_search", "仅当当前页面显示 Back to Search 按钮时返回搜索页；无参数，必须传 {}。"),
    _schema("buy_now", "仅当最新 observation 显示 Buy Now 且商品和规格满足需求时购买；无参数，必须传 {}。"),
    _schema("think", "Record private reasoning.", {"note": {"type": "string"}}, ["note"]),
]
