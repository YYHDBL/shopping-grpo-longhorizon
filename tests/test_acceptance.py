"""验收管线单元测试：硬约束规则、判定合并、Jev 客户端（transport 注入）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.difficulty import difficulty_of
from shopping_grpo.acceptance.hard_checks import parse_budget_limit, run_hard_checks
from shopping_grpo.acceptance.jev_client import JevApiError, JevDecisionsClient
from shopping_grpo.acceptance.pipeline import build_jev_state, combine_verdict


def _record(**overrides):
    base = {
        "asin": "111",
        "title": "测试商品",
        "category": "测试类",
        "shop_name": "测试店",
        "domain_zh": "测试域",
        "tag": "train",
        "pricing": [100.0],
        "attribute": ["a", "b"],
        "customization_options": {
            "颜色": [
                {"value": "红", "price": 100.0, "is_available": True},
                {"value": "蓝", "price": 120.0, "is_available": True},
            ]
        },
        "instructions": [
            {
                "instruction": "买一个测试商品，预算在150元以内。",
                "instruction_options": ["红"],
                "attributes": ["a", "b"],
            }
        ],
    }
    base.update(overrides)
    return base


class TestBudgetParsing:
    def test_budget_below(self):
        assert parse_budget_limit("预算在1000元以下")[0] == 1000.0

    def test_budget_within(self):
        assert parse_budget_limit("价格 500 元以内")[0] == 500.0

    def test_budget_not_exceed(self):
        assert parse_budget_limit("不超过300元的商品")[0] == 300.0

    def test_budget_wan(self):
        assert parse_budget_limit("预算2万以内")[0] == 20000.0

    def test_strictest_wins(self):
        limit, _ = parse_budget_limit("预算在1000元以下，不超过500元")
        assert limit == 500.0

    def test_no_budget(self):
        assert parse_budget_limit("随便买个东西")[0] is None


class TestHardChecks:
    def test_pass(self):
        result = run_hard_checks(_record())
        assert result["status"] == "pass"

    def test_required_option_missing(self):
        record = _record()
        record["instructions"][0]["instruction_options"] = ["绿"]
        result = run_hard_checks(record)
        assert result["status"] == "fail"
        failed = [c for c in result["checks"] if c["status"] == "fail"]
        assert failed[0]["name"] == "required_options"

    def test_required_option_unavailable(self):
        record = _record()
        record["customization_options"]["颜色"][0]["is_available"] = False
        result = run_hard_checks(record)
        assert result["status"] == "fail"

    def test_budget_exceeded_variant_price(self):
        # 必选规格定价 120 元，预算 100 → fail；价格取 variant 而非 pricing[0]。
        record = _record()
        record["instructions"][0]["instruction"] = "买一个测试商品，预算在100元以内。"
        record["instructions"][0]["instruction_options"] = ["蓝"]
        result = run_hard_checks(record)
        assert result["status"] == "fail"
        budget = next(c for c in result["checks"] if c["name"] == "budget")
        assert budget["status"] == "fail"

    def test_structure_missing_title(self):
        result = run_hard_checks(_record(title=""))
        assert result["status"] == "fail"
        assert result["checks"][0]["name"] == "structure"


class TestVerdictCombination:
    def test_hard_fail_short_circuits(self):
        verdict = combine_verdict({"status": "fail", "checks": []}, None)
        assert verdict["verdict"] == "hard_fail"

    def test_jev_fully(self):
        verdict = combine_verdict(
            {"status": "pass"}, {"ok": True, "choice": "fully_satisfies"}
        )
        assert verdict["verdict"] == "accepted"

    def test_jev_partial(self):
        verdict = combine_verdict(
            {"status": "pass"}, {"ok": True, "choice": "partially_satisfies"}
        )
        assert verdict["verdict"] == "semantic_fail"

    def test_jev_insufficient(self):
        verdict = combine_verdict(
            {"status": "pass"}, {"ok": True, "choice": "insufficient_evidence"}
        )
        assert verdict["verdict"] == "unverifiable"

    def test_jev_api_error(self):
        verdict = combine_verdict(
            {"status": "pass"}, {"ok": False, "error": "timeout"}
        )
        assert verdict["verdict"] == "unverifiable"

    def test_level2_missing(self):
        verdict = combine_verdict({"status": "pass"}, None)
        assert verdict["verdict"] == "unverifiable"


class TestJevClient:
    @staticmethod
    def _client(response):
        return JevDecisionsClient(
            api_key="test-key",
            transport=lambda url, payload, headers, timeout: response,
        )

    def test_decide_ok(self):
        response = {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "satisfaction": {
                    "type": "choice",
                    "choice": "fully_satisfies",
                    "probabilities": {"fully_satisfies": 0.97},
                    "confidence": 0.95,
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 1e-5},
            "id": "gen-test",
        }
        result = self._client(response).decide_satisfaction("state")
        assert result["ok"] is True
        assert result["choice"] == "fully_satisfies"
        assert result["model_reported"] == "typesafe/jev-1.13-20260917"
        assert result["request_hash"]

    def test_unexpected_choice_raises(self):
        response = {
            "answers": {"satisfaction": {"type": "choice", "choice": "maybe"}},
        }
        with pytest.raises(JevApiError):
            self._client(response).decide_satisfaction("state")


class TestJevState:
    def test_state_contains_fields_and_no_leak(self):
        state = build_jev_state(_record())
        assert "用户需求" in state
        assert "测试商品" in state
        assert "100.0" in state
        assert "红" in state
        # 不应包含画像、reward 等无关字段。
        assert "persona" not in state.lower()


class TestDifficulty:
    def test_easy(self):
        assert difficulty_of(_record()) == "easy"

    def test_hard_by_attributes(self):
        record = _record()
        record["instructions"][0]["attributes"] = [str(i) for i in range(9)]
        assert difficulty_of(record) == "hard"

    def test_hard_by_combinations(self):
        record = _record()
        record["customization_options"] = {
            f"轴{i}": [
                {"value": str(j), "price": 100.0, "is_available": True}
                for j in range(6)
            ]
            for i in range(2)
        }
        assert difficulty_of(record) == "hard"

    def test_medium(self):
        record = _record()
        record["instructions"][0]["attributes"] = ["a", "b", "c", "d", "e"]
        record["customization_options"] = {
            "颜色": [
                {"value": "红", "price": 100.0, "is_available": True},
                {"value": "蓝", "price": 120.0, "is_available": True},
            ]
        }
        assert difficulty_of(record) == "medium"
