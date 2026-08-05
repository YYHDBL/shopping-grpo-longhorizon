# Repository contract

This repository supports one post-training workflow:

```text
Baseline → SFT → GRPO → Evaluation
```

The original shopping profile remains frozen at ShopSimulator Environment v2.1,
Reward v3, observation v2 and tool schema v2. The parallel long-horizon Feed
profile is versioned independently under ``shopping_grpo.feed`` and must not
change, alias or weaken the original profile's contracts. Do not add compatibility
launchers, historical datasets, old benchmarks, machine-specific paths or
experiment journals.

Shopping-profile training data must never overlap `data/evaluation/tasks.jsonl`.
Its strict success still requires a complete `gold_purchase` terminal result with
`reward_valid=true`. Feed-profile train, validation and frozen evaluation episodes
must be disjoint by both episode and user identifiers, and all generated artifacts
must carry a versioned manifest and content hashes.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.
