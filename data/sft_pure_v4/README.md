# Pure DeepSeek-v4 SFT pool

This directory contains the deduplicated union of the current DeepSeek-v4
teacher set and the new portion of the supplied `merged` package.

- `all.jsonl`: 1,192 clean, unique-task, gold-purchase trajectories.
- `difficulty_labels.jsonl`: intrinsic task difficulty and separate observed
  trajectory-complexity labels from `deepseek-v4-flash`.
- `duplicate_report.json`: source, quality features, and the selected/discarded
  row for every duplicate task.
- `metadata.json`: counts, hashes, labeling provenance, and mix feasibility.

The active SFT recipe keeps the natural 23.8% / 66.9% / 9.2% difficulty mix;
forcing 30% / 50% / 20% would discard valid rows merely because hard examples
are scarce. The deterministic split and cumulative training stages live in
[`../sft_curriculum/`](../sft_curriculum/README.md).
