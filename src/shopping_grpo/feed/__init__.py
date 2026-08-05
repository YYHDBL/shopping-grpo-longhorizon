"""Long-horizon feed-shopping environment contracts.

This package is a parallel profile and does not alter the frozen ShopSimulator runtime.
"""

from shopping_grpo.feed.calibration import BehaviorCalibration, calibrate_behavior
from shopping_grpo.feed.catalog import ProductCatalog, load_embeddings
from shopping_grpo.feed.credit import (
    counterfactual_advantage,
    discounted_returns,
    event_credit_by_source,
)
from shopping_grpo.feed.evaluation import evaluate_episode, evaluate_episodes
from shopping_grpo.feed.model_rollout import rollout_frozen_dataset, rollout_task
from shopping_grpo.feed.policies import (
    PopularPolicy,
    RandomPolicy,
    RulePolicy,
    SimilarityPolicy,
    TeacherPolicy,
    rollout_episode,
)
from shopping_grpo.feed.schema import (
    Decision,
    EpisodeResult,
    EpisodeSeed,
    EventType,
    FeedAction,
    FeedTransition,
    Persona,
    Product,
    Relationship,
    RewardBreakdown,
    Strategy,
    Surface,
    ToolRecord,
    UserEvent,
    Video,
    iter_jsonl,
    load_jsonl,
    read_jsonl,
    write_jsonl,
)
from shopping_grpo.feed.simulator import FeedShoppingEnv
from shopping_grpo.feed.workflow import run_cpu_mvp, verify_cpu_mvp

__all__ = [
    "BehaviorCalibration",
    "Decision",
    "EpisodeResult",
    "EpisodeSeed",
    "EventType",
    "FeedAction",
    "FeedShoppingEnv",
    "FeedTransition",
    "Persona",
    "PopularPolicy",
    "Product",
    "ProductCatalog",
    "RandomPolicy",
    "Relationship",
    "RewardBreakdown",
    "RulePolicy",
    "SimilarityPolicy",
    "Strategy",
    "Surface",
    "TeacherPolicy",
    "ToolRecord",
    "UserEvent",
    "Video",
    "calibrate_behavior",
    "counterfactual_advantage",
    "discounted_returns",
    "evaluate_episode",
    "evaluate_episodes",
    "event_credit_by_source",
    "iter_jsonl",
    "load_jsonl",
    "load_embeddings",
    "read_jsonl",
    "rollout_episode",
    "rollout_frozen_dataset",
    "rollout_task",
    "run_cpu_mvp",
    "verify_cpu_mvp",
    "write_jsonl",
]
