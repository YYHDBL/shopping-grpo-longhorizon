from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
import unittest

from shopping_grpo.feed.credit import (
    counterfactual_advantage,
    discounted_returns,
    event_credit_by_source,
)
from shopping_grpo.feed.evaluation import evaluate_episode, evaluate_episodes
from shopping_grpo.feed.policies import (
    POLICY_REGISTRY,
    PopularPolicy,
    RandomPolicy,
    RulePolicy,
    SimilarityPolicy,
    TeacherPolicy,
    rollout_episode,
)
from shopping_grpo.feed.schema import EpisodeResult, EpisodeSeed, Persona, Product, Video
from shopping_grpo.feed.simulator import FeedShoppingEnv


class _Approx:
    """Small recursive numeric comparator so Feed tests stay dependency-free."""

    def __init__(self, expected, tolerance=1e-9):
        self.expected = expected
        self.tolerance = tolerance

    def __eq__(self, actual):
        if isinstance(self.expected, dict):
            return set(actual) == set(self.expected) and all(
                _Approx(value, self.tolerance) == actual[key]
                for key, value in self.expected.items()
            )
        if isinstance(self.expected, (list, tuple)):
            return len(actual) == len(self.expected) and all(
                _Approx(expected, self.tolerance) == observed
                for observed, expected in zip(actual, self.expected)
            )
        return math.isclose(
            float(actual),
            float(self.expected),
            rel_tol=self.tolerance,
            abs_tol=self.tolerance,
        )


class TinyCatalog:
    def __init__(self, products):
        self.products = {product.product_id: product for product in products}

    def get(self, product_id):
        return self.products.get(product_id)

    def search(self, query, **kwargs):
        del query
        candidate_ids = set(kwargs.get("candidate_ids") or self.products)
        rows = [row for row in self.products.values() if row.product_id in candidate_ids]
        return sorted(
            rows,
            key=lambda row: (-row.popularity, -float(row.rating or 0), row.product_id),
        )[: kwargs.get("limit", 8)]

    def alternatives(self, product_id, limit=5):
        category = self.products[product_id].category
        return [
            row
            for row in self.products.values()
            if row.product_id != product_id and row.category == category
        ][:limit]

    def complements(self, product_id, limit=5):
        ids = set(self.products[product_id].complement_product_ids)
        return [row for row in self.products.values() if row.product_id in ids][:limit]


def fixture():
    products = [
        Product(
            "P1",
            "Minimal storage box",
            "storage",
            25.0,
            attributes=("storage", "minimal"),
            rating=4.8,
            popularity=9.0,
            tags=("scratch-risk",),
            complement_product_ids=("P3",),
        ),
        Product(
            "P2",
            "Budget storage box",
            "storage",
            15.0,
            attributes=("storage",),
            rating=4.2,
            popularity=4.0,
        ),
        Product(
            "P3",
            "Cable clips",
            "cable",
            8.0,
            attributes=("cable",),
            rating=4.5,
            popularity=6.0,
        ),
    ]
    videos = tuple(
        Video(
            f"V{index}",
            caption="Small home-office storage makeover",
            objects=("storage", "cable"),
            style=("minimal",),
            related_product_ids=("P1", "P2"),
            duration_seconds=30.0,
        )
        for index in range(4)
    )
    seed = EpisodeSeed(
        "episode-policy",
        Persona(
            "U1",
            budget=100.0,
            category_interests=("storage",),
            style_preferences=("minimal",),
            price_sensitivity=0.7,
        ),
        videos,
        tuple(product.product_id for product in products),
        31,
        inventory={product.product_id: 10 for product in products},
    )
    return seed, TinyCatalog(products)


class PublicOnlyProxy:
    """Expose exactly the interaction surface a behavior policy is allowed to use."""

    __slots__ = ("wrapped",)

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def observation(self):
        return self.wrapped.observation()

    def call_tool(self, name, arguments=None):
        return self.wrapped.call_tool(name, arguments)

    def step(self, action):
        return self.wrapped.step(action)


POLICIES = (
    RandomPolicy(seed=2, recommendation_probability=1.0, delay_probability=0.0),
    PopularPolicy(),
    SimilarityPolicy(),
    RulePolicy(),
    TeacherPolicy(),
)


def _check_policy_uses_public_interface_and_emits_guard_legal_action(policy):
    seed, catalog = fixture()
    proxy = PublicOnlyProxy(FeedShoppingEnv(seed, catalog))
    action = policy.act(proxy)
    state = proxy.observation()
    assert action["decision"] in {"recommend", "delay", "no_recommend"}
    if action["decision"] == "recommend":
        assert set(action["product_ids"]).issubset(state["visible_product_ids"])
        assert set(action["evidence_ids"]).issubset(state["evidence_ids"])
        assert any(item.startswith("product.") for item in action["evidence_ids"])
        assert any(
            item.startswith(("video.", "history.", "persona."))
            for item in action["evidence_ids"]
        )
        assert all(
            any(product_id in item.split(".") for item in action["evidence_ids"])
            for product_id in action["product_ids"]
        )
        if len(action["product_ids"]) == 2:
            assert action["relationship"] == "bundle"
    proxy.step(action)  # The simulator guard is the final legality oracle.


def _check_invalid_external_teacher_bundle_falls_back_to_legal_policy():
    seed, catalog = fixture()

    def invalid_teacher(context):
        retrieved = context.call_tool("retrieve_products", {"query": "storage"})
        state = context.observation()
        return {
            "decision": "recommend",
            "surface": "bundle",
            "strategy": "bundle",
            "relationship": "primary",
            "product_ids": ["P1", "P2"],
            "evidence_ids": [
                next(item for item in state["evidence_ids"] if item.startswith("video.")),
                retrieved["evidence_ids"][0],
            ],
        }

    proxy = PublicOnlyProxy(FeedShoppingEnv(seed, catalog))
    action = TeacherPolicy(teacher=invalid_teacher).act(proxy)
    self_grounded = all(
        any(product_id in item.split(".") for item in action["evidence_ids"])
        for product_id in action["product_ids"]
    )
    assert action["decision"] in {"recommend", "delay", "no_recommend"}
    assert len(action["product_ids"]) != 2 or action["relationship"] == "bundle"
    assert not action["product_ids"] or self_grounded
    proxy.step(action)


def _check_builtin_teacher_covers_curriculum_action_families():
    seed, catalog = fixture()
    env = FeedShoppingEnv(seed, catalog)
    policy = TeacherPolicy(max_recent_interventions=100)
    strategies = []
    while not env.done:
        action = policy.act(env)
        strategies.append(action["strategy"])
        env.step(action)
    assert {"bundle", "cheaper_alternative", "review_summary", "discount"}.issubset(
        strategies
    )


def _check_policy_module_never_mentions_private_latent_state():
    import shopping_grpo.feed.policies as policies

    assert "_latent" not in inspect.getsource(policies)
    assert set(POLICY_REGISTRY) == {"random", "popular", "similarity", "rule", "teacher"}


def _check_rollout_episode_returns_replayable_typed_trajectory():
    seed, catalog = fixture()
    result = rollout_episode(FeedShoppingEnv(seed, catalog), SimilarityPolicy())
    assert isinstance(result, EpisodeResult)
    assert result.done
    assert len(result.transitions) == len(seed.videos)
    assert result.transitions[-1].done
    assert all("raw_events" in transition.metadata for transition in result.transitions)
    assert result.total_reward == _Approx(
        sum(transition.reward.total for transition in result.transitions)
    )


def _check_discounted_returns_and_delayed_event_credit():
    assert discounted_returns([1.0, 2.0, 3.0], gamma=0.5) == _Approx(
        [2.75, 3.5, 3.0]
    )
    trajectory = [
        {
            "step": 0,
            "action": {"decision": "recommend"},
            "events": [],
            "reward": {"engagement": 1.0, "total": 1.0},
        },
        {
            "step": 1,
            "action": {"decision": "no_recommend"},
            "events": [{"event_type": "purchase", "step": 1, "source_step": 0}],
            "reward": {"qualified_purchase_value": 4.0, "total": 4.0},
        },
    ]
    assert event_credit_by_source(trajectory) == _Approx({0: 5.0, 1: 0.0})
    assert event_credit_by_source(
        [{"event_type": "return", "step": 3, "source_step": 1, "reward": -2.0}],
        gamma=0.5,
    ) == _Approx({1: -0.5})
    assert event_credit_by_source(
        [
            {
                "step": 0,
                "reward": {"total": 1.0},
                "events": [],
            },
            {
                "step": 1,
                "reward": {
                    "qualified_purchase_value": 2.0,
                    "purchase_satisfaction": 1.0,
                    "total": 3.0,
                },
                "events": [{"event_type": "purchase", "step": 1, "source_step": 0}],
            },
        ]
    ) == _Approx({0: 4.0, 1: 0.0})


def _check_crn_counterfactual_replaces_only_target_and_rebuilds_later_evidence():
    seed, catalog = fixture()
    calibration = SimpleNamespace(
        coefficients={"click_bias": 12.0, "cart_bias": 12.0, "buy_bias": 12.0}
    )

    def factory():
        return FeedShoppingEnv(seed, catalog, calibration=calibration)

    factual = rollout_episode(factory(), TeacherPolicy())
    target = next(
        transition.step
        for transition in factual.transitions
        if transition.action.decision.value == "recommend"
    )
    comparison = counterfactual_advantage(factory, factual, target)
    assert math.isfinite(comparison["A_cf"])
    cf_transitions = comparison["counterfactual_episode"]["transitions"]
    assert cf_transitions[target]["action"]["decision"] == "no_recommend"
    for index, (recorded, replayed) in enumerate(zip(factual.transitions, cf_transitions)):
        if index == target or recorded.action.decision.value != "recommend":
            continue
        # Divergent purchases may make a later product ineligible; otherwise the
        # replay preserves the factual action and reconstructs current evidence.
        if replayed["action"]["decision"] == "recommend":
            assert replayed["action"]["product_ids"] == list(recorded.action.product_ids)
            assert replayed["action"]["evidence_ids"]

    factual_events = comparison["factual_episode"]["transitions"][target]["metadata"][
        "raw_events"
    ]
    counterfactual_events = cf_transitions[target]["metadata"]["raw_events"]
    content_types = {"watch", "skip", "like"}
    factual_content = [
        (event["event_type"], event.get("value"))
        for event in factual_events
        if event["event_type"] in content_types
    ]
    counterfactual_content = [
        (event["event_type"], event.get("value"))
        for event in counterfactual_events
        if event["event_type"] in content_types
    ]
    assert factual_content == counterfactual_content

    delay = {
        "decision": "delay",
        "surface": "none",
        "strategy": "none",
        "relationship": "primary",
        "product_ids": [],
        "evidence_ids": [],
        "explanation": "Exact CRN delay replacement.",
    }
    delayed = counterfactual_advantage(
        factory,
        factual,
        target,
        replacement_action=delay,
    )
    assert delayed["counterfactual_action"] == delay
    assert (
        delayed["counterfactual_episode"]["transitions"][target]["action"]["decision"]
        == "delay"
    )


def synthetic_episode():
    context = "video.V.context"
    product = "product.P.retrieved"

    def transition(step, action, events, reward):
        return {
            "step": step,
            "action": action,
            "events": events,
            "reward": reward,
            "tool_records": (
                [{"tool_name": "retrieve_products"}]
                if action["decision"] == "recommend"
                else []
            ),
            "metadata": {
                "raw_events": events,
                "raw_reward_breakdown": reward,
            },
        }

    recommend = {
        "decision": "recommend",
        "surface": "product_card",
        "strategy": "direct",
        "relationship": "primary",
        "product_ids": ["P"],
        "evidence_ids": [context, product],
    }
    return {
        "episode_id": "metrics-1",
        "total_reward": 7.0,
        "final_state": {
            "evaluation_summary": {
                "net_revenue": 20.0,
                "terminal_fatigue": 0.4,
                "terminal_satisfaction": 0.7,
            }
        },
        "transitions": [
            transition(
                0,
                recommend,
                [
                    {"event_type": "watch", "step": 0, "value": 10.0},
                    {"event_type": "like", "step": 0},
                    {"event_type": "click", "step": 0, "product_id": "P"},
                    {"event_type": "cart", "step": 0, "product_id": "P"},
                ],
                {"total": 1.0},
            ),
            transition(
                1,
                {
                    "decision": "no_recommend",
                    "surface": "none",
                    "strategy": "none",
                    "product_ids": [],
                    "evidence_ids": [],
                },
                [
                    {"event_type": "skip", "step": 1, "value": 2.0},
                    {
                        "event_type": "no_recommend",
                        "step": 1,
                        "metadata": {"hindsight_correct": True},
                    }
                ],
                {"correct_no_recommend": 0.08, "total": 0.08},
            ),
            transition(
                2,
                {**recommend, "evidence_ids": []},
                [
                    {"event_type": "watch", "step": 2, "value": 8.0},
                    {
                        "event_type": "bundle_offer",
                        "step": 2,
                        "metadata": {"complementary": True},
                    },
                    {"event_type": "repeat_exposure", "step": 2, "product_id": "P"},
                    {
                        "event_type": "purchase",
                        "step": 2,
                        "source_step": 0,
                        "product_id": "P",
                        "value": 20.0,
                        "metadata": {"qualified": True},
                    },
                ],
                {"irrelevant_recommendation": -0.3, "total": 2.0},
            ),
            transition(
                3,
                {**recommend, "product_ids": ["Q"]},
                [
                    {"event_type": "watch", "step": 3, "value": 12.0},
                    {"event_type": "click", "step": 3, "product_id": "Q"},
                    {"event_type": "cart", "step": 3, "product_id": "Q"},
                    {
                        "event_type": "purchase",
                        "step": 3,
                        "source_step": 3,
                        "product_id": "Q",
                        "value": 30.0,
                        "metadata": {"qualified": False},
                    },
                    {
                        "event_type": "return",
                        "step": 3,
                        "source_step": 3,
                        "product_id": "Q",
                        "value": 30.0,
                    },
                ],
                {"return_penalty": -1.0, "total": 3.92},
            ),
        ],
    }


def _check_evaluation_reports_conversion_experience_grounding_and_value_metrics():
    metrics = evaluate_episode(synthetic_episode())
    assert metrics["qualified_purchase_rate"] == _Approx(0.5)
    assert metrics["correct_no_recommend_rate"] == _Approx(1.0)
    assert metrics["interventions_per_100"] == _Approx(75.0)
    assert metrics["return_rate"] == _Approx(0.5)
    assert metrics["irrelevant_recommendation_rate"] == _Approx(1 / 3)
    assert metrics["repeat_exposure_rate"] == _Approx(1 / 3)
    assert metrics["net_revenue"] == _Approx(20.0)
    assert metrics["fatigue"] == _Approx(0.4)
    assert metrics["grounded_recommendation_rate"] == _Approx(2 / 3)
    assert metrics["long_term_return"] == _Approx(7.0)
    assert metrics["click_rate"] == _Approx(2 / 3)
    assert metrics["add_to_cart_rate"] == _Approx(2 / 3)
    assert metrics["purchase_rate"] == _Approx(2 / 3)
    assert metrics["watches"] == 3
    assert metrics["skips"] == 1
    assert metrics["likes"] == 1
    assert metrics["mean_dwell_seconds"] == _Approx(8.0)
    assert metrics["skip_rate"] == _Approx(0.25)
    assert metrics["terminal_satisfaction"] == _Approx(0.7)
    assert metrics["bundle_offers"] == 1
    assert metrics["complementary_bundle_precision"] == _Approx(1.0)
    assert metrics["unsupported_claims"] == 1
    assert metrics["unsupported_claim_rate"] == _Approx(1 / 3)
    assert metrics["tool_calls"] == 3
    assert metrics["mean_tool_calls_per_step"] == _Approx(0.75)

    aggregate = evaluate_episodes([synthetic_episode(), synthetic_episode()])
    assert aggregate["episodes"] == 2
    assert aggregate["purchases"] == 4
    assert aggregate["long_term_return"] == _Approx(7.0)
    assert aggregate["mean_terminal_satisfaction"] == _Approx(0.7)


class FeedPoliciesCreditEvaluationTest(unittest.TestCase):
    def test_all_policies_use_only_public_interface(self):
        for policy in POLICIES:
            with self.subTest(policy=type(policy).__name__):
                _check_policy_uses_public_interface_and_emits_guard_legal_action(policy)

    def test_policy_module_never_mentions_private_latent_state(self):
        _check_policy_module_never_mentions_private_latent_state()

    def test_invalid_external_teacher_bundle_falls_back(self):
        _check_invalid_external_teacher_bundle_falls_back_to_legal_policy()

    def test_builtin_teacher_covers_curriculum_action_families(self):
        _check_builtin_teacher_covers_curriculum_action_families()

    def test_rollout_episode_returns_replayable_typed_trajectory(self):
        _check_rollout_episode_returns_replayable_typed_trajectory()

    def test_discounted_returns_and_delayed_event_credit(self):
        _check_discounted_returns_and_delayed_event_credit()

    def test_crn_counterfactual_replay(self):
        _check_crn_counterfactual_replaces_only_target_and_rebuilds_later_evidence()

    def test_evaluation_metrics(self):
        _check_evaluation_reports_conversion_experience_grounding_and_value_metrics()


if __name__ == "__main__":
    unittest.main()
