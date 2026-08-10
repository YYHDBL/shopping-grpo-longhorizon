# Training Diagnostics Implementation Plan

**Goal:** Preserve enough SFT and GRPO evidence to compare checkpoints and diagnose low RL gains without changing the learning objective.

**Design:** Reuse Transformers checkpoint logs and the existing veRL/Shopping metrics. Add public action and guard summaries to each rollout, append generation-group and optimizer metrics to one JSONL file, enable entropy measurement, and retain one SFT checkpoint per default epoch.

## Tasks

1. Add CPU tests for rollout diagnostics, JSONL append behavior, launcher wiring, entropy configuration, and SFT checkpoint retention.
2. Implement the smallest project-side JSONL helper and expose missing rollout fields.
3. Wire the pinned veRL patch to write generation and optimizer events beside the GRPO output.
4. Update defaults/docs, then run focused CPU checks and verify the patch against pinned veRL source when available.

