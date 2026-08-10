# LoRA SFT

## Purpose

The base model can speak naturally but does not reliably follow
ShopSimulator's action protocol. Supervised fine-tuning teaches the basic
policy: issue legal tool calls, use observations as evidence, select product
variants and terminate.

## Inputs

- Base model: `Qwen/Qwen3.5-2B`
- Train data: `data/sft/train.jsonl` (800 rows)
- Validation data: `data/sft/validation.jsonl` (200 rows)
- Target: assistant tokens only; user and tool-observation tokens are masked

The data provenance and hashes are recorded in
[`data/sft/metadata.json`](../data/sft/metadata.json).

## Run

After `bash scripts/setup.sh`:

```bash
bash scripts/sft.sh
```

The launcher trains a LoRA adapter and then merges it with the base model:

```text
outputs/models/sft-lora/
outputs/models/sft-merged/
```

Default recipe:

| Setting | Value |
|---|---|
| Maximum sequence length | 24,576 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | `1e-4` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient checkpointing | enabled |
| Attention implementation | SDPA |
| Saved epoch checkpoints | 3 |

The long context is intentional: a training example includes the complete
multi-turn interaction. Shortening it may truncate the terminal decision or the
evidence that supports it.

Each `checkpoint-*` directory keeps its adapter and `trainer_state.json`; the
root `train_summary.json` also contains the complete `log_history`. Keep all
three checkpoints until the same held-out task suite has been run against each
one. Validation loss is useful for screening, but the checkpoint used by GRPO
should be selected by held-out task success and failure-type coverage.

## Evaluate

```bash
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

The reported checkpoint completed 141 optimizer steps. Its validation loss was
0.3365 after epoch 1, 0.3189 after epoch 2 and 0.3147 after epoch 3. The frozen
result and reproduction config are in [`experiments/sft/`](../experiments/sft/).

## Output contract

GRPO starts from the merged model, not directly from the adapter:

```text
GRPO_MODEL_PATH=outputs/models/sft-merged
```

This boundary keeps the GRPO launcher independent of the SFT trainer process.
