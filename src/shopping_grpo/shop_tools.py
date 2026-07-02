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
            },
        },
    }


SHOP_TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "Search products by query.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema("open_product", "Open a product by ASIN.", {"asin": {"type": "string"}}, ["asin"]),
    _schema("select_option", "Select a product option.", {"value": {"type": "string"}}, ["value"]),
    _schema("view_description", "Open the description page."),
    _schema("view_features", "Open the features page."),
    _schema("view_reviews", "Open the reviews page."),
    _schema("view_attributes", "Open the attributes page."),
    _schema("next_page", "Go to the next search page."),
    _schema("prev_page", "Go to the previous search page."),
    _schema("back_to_search", "Return to search page."),
    _schema("buy_now", "Buy the currently selected product."),
    _schema("think", "Record private reasoning.", {"note": {"type": "string"}}, ["note"]),
]
