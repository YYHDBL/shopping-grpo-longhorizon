"""Check component isolation and catalog/index pairing without model execution."""

import hashlib
import json
import sqlite3

import pytest

from scripts.build_component_profile import ROOT, compose, verify_catalog_pair
from shopping_grpo.environment.interface_support import (
    render_exact_observation,
    resolve_option_call,
)


@pytest.fixture
def catalog(tmp_path):
    products = tmp_path / "products.json"
    products.write_text("[]")
    index = tmp_path / "index.sqlite"
    with sqlite3.connect(index) as conn:
        conn.execute("CREATE TABLE manifest(payload TEXT)")
        conn.execute(
            "INSERT INTO manifest VALUES (?)",
            (
                json.dumps(
                    {"product_data_sha256": hashlib.sha256(products.read_bytes()).hexdigest()}
                ),
            ),
        )
    return products, index


def test_profile_switches_are_independent(catalog, tmp_path):
    original_tools = json.loads((ROOT / "configs/tools.json").read_text())
    full = compose([], *catalog, tmp_path / "full")
    for component in "ABCDEFG":
        agent, tools, service, protocol = compose([component], *catalog, tmp_path / component)
        assert [t["tool_schema"] for t in tools["tools"]] == [
            t["tool_schema"] for t in original_tools["tools"]
        ]
        assert tools["tools"][0]["config"]["purchase_checks"] == (component != "B")
        assert tools["tools"][0]["config"]["interface_support"] == (component != "C")
        assert ("system_prompt_file" in agent[0]) == (component != "A")
        assert agent[0]["observation_token_budget"] == (1536 if component == "D" else 8192)
        assert agent[0]["context_compaction_enable"] == (component != "E")
        assert ("--reviewed-reward" in service["argv"]) == (component != "G")
        assert protocol["catalog_role"] == ("original" if component == "F" else "reviewed")
        assert not protocol["automatic_details"]
        for other in set("ABCDEFG") - {component}:
            assert protocol["components"][other] == full[3]["components"][other]


def test_catalog_mismatch_rejected_before_output(catalog, tmp_path):
    products, index = catalog
    products.write_text('[{"changed": true}]')
    with pytest.raises(ValueError, match="SHA-256"):
        compose([], products, index, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_missing_index_is_not_created(catalog, tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        verify_catalog_pair(catalog[0], tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_exact_labels_and_conservative_resolution():
    state = {
        "observation_version": "shopping-observation-v2",
        "page_type": "product_detail",
        "product": {"asin": "123456789012", "title": "mug"},
        "available_options": {"size": ["L  size"]},
        "actions": ["L  size", "Buy Now"],
    }
    observation = render_exact_observation(state)
    assert '"L  size"' in observation.split("可点击的按钮: ")[1]
    call = {"function": {"name": "select_option", "arguments": {"value": "L size"}}}
    resolved, audit = resolve_option_call(call, observation)
    assert json.loads(resolved["function"]["arguments"])["value"] == "L  size"
    assert audit["method"] == "unique_whitespace_match"
    assert call["function"]["arguments"]["value"] == "L size"
    state["available_options"]["size"].append("L size")
    state["actions"].append("L size")
    call["function"]["arguments"]["value"] = "Lsize"
    assert resolve_option_call(call, render_exact_observation(state))[1] is None
    state["target_asin"] = "123456789012"
    with pytest.raises(ValueError, match="forbidden"):
        render_exact_observation(state)
