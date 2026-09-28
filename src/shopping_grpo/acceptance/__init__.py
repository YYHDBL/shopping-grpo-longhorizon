"""V2.0 数据验收管线：第一级本地硬约束 + 第二级 Jev 语义验收。"""

from shopping_grpo.acceptance.hard_checks import run_hard_checks, HARD_CHECKS_VERSION
from shopping_grpo.acceptance.jev_client import JevDecisionsClient, JEV_RUBRIC_VERSION
from shopping_grpo.acceptance.pipeline import run, combine_verdict

__all__ = [
    "run",
    "run_hard_checks",
    "combine_verdict",
    "JevDecisionsClient",
    "HARD_CHECKS_VERSION",
    "JEV_RUBRIC_VERSION",
]
