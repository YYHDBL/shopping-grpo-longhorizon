# LoRA SFT

## Purpose

The base model can speak naturally but does not reliably follow
ShopSimulator's action protocol. Supervised fine-tuning teaches the basic
policy: issue legal tool calls, use observations as evidence, select product
variants and terminate.

## Inputs

- Base model: `Qwen/Qwen3.5-2B`
- Main data: `data/sft_pure_v4/all.jsonl` (1,192 rows)
- Fixed curriculum manifest: `data/sft_curriculum/manifest.json`
- Gradient rows: 1,073; development rows: 119; Final evaluation overlap: 0
- Target: assistant tokens only; user and tool-observation tokens are masked

The source and label hashes, exact task IDs, stage definitions, and review-only
flags are frozen in the curriculum manifest. The older `data/sft/` split is
kept only for reproducing the historical baseline.

## Run

After `bash scripts/setup.sh`:

```bash
# Check all six train/merge commands without loading a model.
bash scripts/sft_curriculum.sh --dry-run

# Run A -> B -> C on the server.
bash scripts/sft_curriculum.sh --swanlab
```

The launcher trains a LoRA adapter and then merges it with the base model:

```text
outputs/models/sft-curriculum/stage-a/{adapter,merged}/
outputs/models/sft-curriculum/stage-b/{adapter,merged}/
outputs/models/sft-curriculum/stage-c/{adapter,merged}/
```

Default recipe:

| Setting | Value |
|---|---|
| Maximum sequence length | 24,576 |
| Epochs | 1 per stage |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | `1e-4` -> `7e-5` -> `5e-5` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient checkpointing | enabled |
| Attention implementation | SDPA |
| Saved epoch checkpoints | 3 |

The long context is intentional: a training example includes the complete
multi-turn interaction. Shortening it may truncate the terminal decision or the
evidence that supports it.

Stage A learns the action protocol from 256 foundation rows. Stage B restarts a
fresh LoRA on A's merged checkpoint and uses 799 cumulative constraint rows.
Stage C does the same from B and uses all 1,073 training rows. Therefore simple
skills receive three passes, constraint handling two, and long-horizon strategy
one. Use `--start-stage b` after A is complete, or `--stop-after-stage b` for a
bounded server run. A checkpoint interrupted inside a stage can be resumed
with `--start-stage <stage> --resume-from-checkpoint <checkpoint-dir>`.

## Evaluate

```bash
bash scripts/serve_model.sh outputs/models/sft-curriculum/stage-c/merged
bash scripts/evaluate.sh sft
```

Validation loss is a training-health signal, not the final model score. Select
among stages using the 119-row development split and failure-type coverage.
Run Final-200 Clean only after the recipe is frozen, so the final benchmark is
not silently used for checkpoint selection.

## Output contract

GRPO starts from the merged model, not directly from the adapter:

```text
GRPO_MODEL_PATH=outputs/models/sft-curriculum/stage-c/merged
```

This boundary keeps the GRPO launcher independent of the SFT trainer process.
