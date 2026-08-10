# Pure DeepSeek-v4 SFT pool

This directory contains the deduplicated union of the current DeepSeek-v4
teacher set and the new portion of the supplied `merged` package.

- `all.jsonl`: 1,192 clean, unique-task, gold-purchase trajectories.
- `difficulty_labels.jsonl`: intrinsic task difficulty and separate observed
  trajectory-complexity labels from `deepseek-v4-flash`.
- `duplicate_report.json`: source, quality features, and the selected/discarded
  row for every duplicate task.
- `metadata.json`: counts, hashes, labeling provenance, and mix feasibility.

The requested 30% simple / 50% medium / 20% hard split is intentionally not
materialized yet. The observed pool has only 110 hard rows, so an exact
no-repeat 3:5:2 subset would retain just 550 of 1,192 rows. Prefer retaining the
natural 23.8% / 66.9% / 9.2% mix for the first SFT run, or use sampling weights
at training time if an exact exposure ratio is required.
