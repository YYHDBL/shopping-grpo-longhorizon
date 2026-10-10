"""The G switch affects purchase and candidate eligibility together."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.parametrize("enabled", [False, True])
def test_reviewed_service_boundary(monkeypatch, enabled):
    from web_agent_site.engine import reviewed_reward

    package = ModuleType("web_agent_site.envs")
    monkeypatch.setitem(sys.modules, "web_agent_site.envs", package)

    original_purchase, original_eligibility = object(), object()
    environment = ModuleType("web_agent_site.envs.web_agent_text_env")
    environment.evaluate_purchase = original_purchase
    environment.evaluate_candidate_eligibility = original_eligibility
    api = ModuleType("pack_api")
    monkeypatch.setitem(sys.modules, "pack_api", api)
    monkeypatch.setitem(sys.modules, environment.__name__, environment)
    monkeypatch.setattr(package, "web_agent_text_env", environment, raising=False)
    monkeypatch.setenv("SHOPSIM_PUBLIC_SUPPORT", "1")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["serve"] + (["--reviewed-reward"] if enabled else []))
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/serve_reviewed_environment.py")
    )
    import os

    assert os.environ["SHOPSIM_PUBLIC_SUPPORT"] == "0"
    assert environment.evaluate_purchase is (
        reviewed_reward.evaluate_purchase if enabled else original_purchase
    )
    assert environment.evaluate_candidate_eligibility is (
        reviewed_reward.evaluate_candidate_eligibility if enabled else original_eligibility
    )


def test_reviewed_purchase_retains_complete_original_result():
    from web_agent_site.engine import reviewed_reward, reward

    product = {"asin": "123456789012", "title": "cup", "price": 5}
    goal = {"asin": "123456789012", "instruction_text": "cup"}
    original = reward.evaluate_purchase(product, goal, selected_options={}, price=5)
    reviewed = reviewed_reward.evaluate_purchase(product, goal, selected_options={}, price=5)
    assert reviewed.evidence["legacy_result"] == original.to_dict()
