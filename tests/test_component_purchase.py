from shopping_grpo.environment.purchase_support import purchase_check


def page(price="50", selected="{}", available="{}"):
    return "\n".join(
        [
            "page_type: product_detail",
            f"price: {price}",
            f"selected_options: {selected}",
            f"available_options: {available}",
        ]
    )


def test_purchase_prerequisites_and_explicit_bounds():
    assert purchase_check("预算不超过100元", page("101"))["reason"]
    assert purchase_check("预算不超过100元", page("unknown"))["reason"] == "variant_price_unknown"
    assert (
        purchase_check("买杯子", page(available='{"颜色":["红"]}'))["reason"]
        == "variant_axes_unselected"
    )
    assert purchase_check("预算不超过100元", page("100"))["reason"] is None


def test_no_semantic_requirement_or_hidden_target_check():
    assert purchase_check("需要红色陶瓷杯", page())["reason"] is None
    assert purchase_check("价格100元左右", page("101"))["reason"] is None
