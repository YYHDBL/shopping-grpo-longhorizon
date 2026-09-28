# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Reward v3, observation
v2 and tool schema v2. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

## Evaluation admission (strict success)

Formal evaluation strict success requires a complete `gold_purchase` terminal
result with `reward_valid=true`. This definition is shared by the frozen
evaluation set, the Reward v3 terminal accounting, and the reported metrics.

## SFT training-data admission

SFT trajectories enter the corpus when their terminal result is
`reward_valid=true` and the purchase is either a complete `gold_purchase` or an
alternative purchase approved by the frozen JEV rubric (`fully_satisfies`).
Each accepted alternative keeps its JEV verdict and the full terminal result
for audit. Evaluation admission stays stricter than SFT admission by design:
alternatives never count as strict success in formal metrics.

## Data separation

Training data must never overlap the V2 evaluation split. The legacy Final-200
benchmark is retired from V2 (kept in Git history only); the 63 records
historically shared with the SFT corpus are migration-audit references only
and must not enter any active evaluation. The V2 evaluation set is built from
the frozen `tag=eval` split with strict deduplication by stable task id.

Do not start training, merge models or run the formal evaluation unless the
user explicitly requests execution.
