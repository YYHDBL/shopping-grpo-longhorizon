# Data collection

## Goal

The SFT stage needs complete examples of a shopping agent using tools correctly:
searching, opening products, inspecting evidence, choosing options and ending
with a valid purchase. The repository contains the accepted action-only
trajectories, not historical failed collection attempts.

## How the dataset was produced

The current collection used ShopSimulator Environment v2.1, Reward v3 and
`deepseek-v4-flash` as the teacher. It produced 2,498 raw trajectories.
Every trajectory executed its actions in ShopSimulator during collection. The
saved result was accepted only when Environment v2.1 returned a valid Reward v3
gold purchase; no second model judged whether the trajectory succeeded.

Collection audit:

| Item | Value |
|---|---:|
| Raw trajectories | 2,498 |
| Accepted trajectories | 1,026 |
| Acceptance rate | 41.07% |
| Frozen rows used | 1,000 |
| Unused accepted rows | 26 |

The frozen subset was split into 800 training and 200 validation rows. The two
splits are task-disjoint and also have zero task-ID overlap with GRPO and the
Final-200 evaluation set.

## Frozen deliverables

| File | Rows | SHA-256 |
|---|---:|---|
| `data/sft/train.jsonl` | 800 | `8c3a6ff0033f6ea672af609891e747d60652ddc17e8d3c8eacb19e9d96dd9477` |
| `data/sft/validation.jsonl` | 200 | `9525cc2fb04a1d8d38ae2db959397da908dde3fea766f580fdcf77d1239533cc` |

Raw teacher responses are intentionally not committed; their collection path
is retained in `data/sft/metadata.json` as provenance.

## Run a new collection

Start ShopSimulator, configure an OpenAI-compatible Teacher endpoint, and run:

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your-key

python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --model deepseek-v4-flash \
  --target-accepted 1000 \
  --workers 4
```

`raw.jsonl` is the resumable source of truth. Running the same command again
skips completed task attempts and rebuilds all derived files:

```text
outputs/sft-collection/
  raw.jsonl           complete Teacher responses and environment results
  accepted.jsonl      strict Reward v3 gold trajectories
  rejected.jsonl      task IDs and deterministic rejection reasons
  reject_stats.json   aggregate acceptance audit
  sft.jsonl           sanitized training rows before splitting
  train.jsonl         task-disjoint training split
  validation.jsonl    task-disjoint validation split
  metadata.json       row counts, configuration and SHA-256 hashes
```

The command removes all task IDs listed in `data/evaluation/tasks.jsonl` before
collection and checks again while building artifacts. It also keeps at most one
accepted trajectory per task. To rebuild the derived files without contacting
the Teacher or environment, run:

```bash
python scripts/collect_sft_data.py \
  --build-only \
  --output-dir outputs/sft-collection
```

Only copy `train.jsonl`, `validation.jsonl` and their metadata into `data/sft/`
after reviewing the collection audit. Raw Teacher responses remain in
`outputs/` and should not be committed.

## What a training row contains

Each JSONL row is a chat trajectory with:

- the shopping instruction;
- assistant tool calls;
- ShopSimulator tool observations;
- the final terminal action;
- metadata tying the row to Environment v2.1 and Reward v3.

During SFT, user and tool tokens are masked. Loss is computed only on assistant
actions. See [SFT](sft.md) for the exact training recipe.
