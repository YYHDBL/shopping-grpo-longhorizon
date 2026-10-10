import pytest

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


@pytest.mark.parametrize("field_name", ["available_options", "selected_options"])
@pytest.mark.parametrize(
    "value",
    [None, "", "null", "[]", "false", '"unknown"', '{"颜色":', '{"颜色": ["红"]} [TRUNCATED]'],
)
def test_missing_or_unreadable_option_state_blocks_purchase(field_name, value):
    observation = page()
    lines = [line for line in observation.splitlines() if not line.startswith(field_name + ":")]
    if value is not None:
        lines.append(field_name + ": " + value)
    result = purchase_check("买杯子", "\n".join(lines))
    assert result["reason"] == "variant_state_unreadable"
    assert "购买未执行" in result["feedback"]


@pytest.mark.parametrize(
    "available,selected",
    [
        ('{"颜色": null}', "{}"),
        ('{"颜色": "红"}', "{}"),
        ('{"颜色": [null]}', "{}"),
        ('{"颜色": [""]}', "{}"),
        ('{"": ["红"]}', "{}"),
        ('{"颜色": ["红"]}', '{"颜色": null}'),
        ('{"颜色": ["红"]}', '{"颜色": []}'),
        ('{"颜色": ["红"]}', '{"颜色": "蓝"}'),
        ("{}", '{"颜色": "红"}'),
    ],
)
def test_malformed_or_inconsistent_option_maps_block_purchase(available, selected):
    assert (
        purchase_check("买杯子", page(available=available, selected=selected))["reason"]
        == "variant_state_unreadable"
    )


def test_explicit_empty_options_and_complete_selection_allow_purchase():
    assert purchase_check("买杯子", page(available="{}", selected="{}"))["reason"] is None
    assert (
        purchase_check("买杯子", page(available='{"颜色":["红"]}', selected='{"颜色":"红"}'))[
            "reason"
        ]
        is None
    )


def test_empty_field_does_not_consume_next_line():
    observation = "price: 50\navailable_options: \n{}\nselected_options: {}"
    assert purchase_check("买杯子", observation)["reason"] == "variant_state_unreadable"
