# Optional enhanced shopping harness

The baseline entrypoints remain available. This package brings the previously
separate interaction, public-evidence, scoring-review and runtime-audit
components into the repository. None of the examples below authorize or
automatically start training or evaluation.

## Components and switches

| Component | Entry point | Default |
|---|---|---|
| Current-page tool filtering and purchase prerequisites | evaluation CLI `--public-support` | off |
| Explicit numeric CNY purchase bounds only | `--budget-guard` | off |
| Exact option labels instead of page-local IDs | `--option-labels` with public support | off |
| Strict verification prompt | `--system-prompt configs/enhanced/system_prompt.txt` | upstream prompt |
| Public details / option-image evidence | reviewed environment service flags | off |
| Reviewed scoring | service `--reviewed-reward` | original Reward v3 |
| Original purchase score from reviewed result | evaluation `--reward-policy original` | environment score |
| veRL interaction adapter | enhanced agent/tool configuration | baseline configuration |
| veRL original-score selection | enhanced agent YAML `select_original_reward` | false |
| Trajectory token recording | enhanced agent `trajectory_directory` | no output directory |
| Checkpoint/device attestation, no-update guard | explicit worker setup hook | not installed |

Button labels now retain exact whitespace; illegal search feedback explains how
to return to a searchable page. These two interface corrections apply without
the enhancement flags. Public detail additions and behavioral policies are
optional. The runtime manifest is refreshed to match the packaged source;
catalog contents and task lists are unchanged.

## OpenAI-compatible collector

Use the existing `scripts/evaluate_shop_benchmark.py` with its normal required
model, benchmark and output arguments. Add the desired switches separately.
For the full collector interaction profile, add:

```text
--public-support --option-labels --budget-guard
--system-prompt configs/enhanced/system_prompt.txt
--context-compaction
--observation-token-budget 8192
--observation-detail-token-budget 8192
--observation-generic-token-budget 8192
```

Use a fresh output JSONL for a different protocol. Collection resumes by task ID
and attempt, so reusing a completed output would not evaluate the new setting.
The summary records flags, prompt SHA256 and reward policy. The default
per-call generation limit of this collector is unchanged; this is not the same
generation path as veRL.

## veRL

Select `configs/enhanced/agent_loop.yaml` and `configs/enhanced/tools.json`
through the existing training launcher `--agent-config` and `--tool-config`
options or the corresponding veRL configuration fields. Run from the repository
root so the enhanced system-prompt path resolves. No training dataset is
rewritten; the explicit enhanced adapter replaces the system message before
generation and adds the initial public observation.

The enhanced tools resolve current-page options, check purchases and return
recovery feedback. They retain the original fixed tool schema. Dynamic
current-page schema filtering is implemented in the OpenAI-compatible collector
only; do not describe the two backends as identical.

The enhanced YAML uses 8192 observation budgets, a 22016 input budget, a 24576
context window, history compaction and a 2048 generation reserve. The reserve
is not a per-turn output cap. The framework retains ownership of sampling and
the total response budget; this change does not add a 2048-token turn limit.

The strict prompt can cause repetitive indecision. It is an explicit profile,
not a proven cure for looping. Changing prompt, observation budgets and action
handling together is a bundle intervention, not a component ablation.

For reviewed environments, `select_original_reward: true` selects the stored
original purchase score and retains review diagnostics separately. Missing
legacy purchase evidence is an error. Nonpurchase failures retain the upstream
reward; this adapter does not add a failure-penalty policy.

## Public evidence and reviewed environment

`scripts/serve_reviewed_environment.py` uses the existing ShopSimulator routes
and loopback binding. Optional arguments are `--public-details`,
`--option-evidence PATH`, `--reviewed-reward`, and `--products PATH`.
The normal environment launcher remains unchanged. Use the installed
ShopSimulator interpreter and dependencies.

Only whitelisted public description, feature, option-price and evidence fields
are added to observations. Evidence manifests use version
`public-option-evidence-v1` and bind each record to an exact ASIN, axis,
option value and HTTPS image URL; records contain `caption` and `facts`.
The environment performs no image download or model call. Record the source
and review provenance when supplying a manifest. This repository does not
bundle the historical catalog, images or manual task-specific annotations.

The review modules preserve the original score in `evidence.legacy_result`.
They distinguish unknown evidence from contradiction and implement limited
reviewed aliases, quantity/payment checks and public-budget reasoning.
Approximate budget diagnostics are sensitivity checks, not invented ground
truth tolerances. Scoring review is separate from changes to actor behavior.
Use one explicitly selected policy for paired comparisons.

## Data quality and offline comparisons

`python -m shopping_grpo.environment.task_data_quality --help` exposes the
existing audit and reviewed-repair utility. Repairs require explicit source
hash, scope and before/after fields; output must be a new file. It never
activates the repaired catalog automatically. Missing requirements must remain
unresolved rather than being filled from hidden targets. Historical field-swap
repairs are not silently applied to repository data.

`python -m shopping_grpo.evaluation.paired_audit --help` compares existing
collector JSONL files without model calls. It requires unique, identical task
sets, complete purchase terminals and valid gold-purchase rewards. Original
scoring is the default; input hashes and gained/lost task IDs are written to a
new report. Sampling repetitions must be separated into paired attempts before
using this one-attempt utility. It does not convert veRL token dumps into
collector JSONL or claim statistical significance.

## Runtime checks

SFT writes tokenizer metadata and summaries only on the world-zero process.
GRPO accepts explicit manifest/agent/tool/config paths and preserves caller
PYTHONPATH additions. A local Transformers source archive requires an explicit
`SHOPPING_TRANSFORMERS_ARCHIVE_SHA256` in addition to the expected revision;
there is no built-in workstation archive hash.

The optional training worker hook
shopping_grpo.training.grpo.adapter.update_audit.install records before/after
parameter fingerprints and per-rank device selection to
SHOPPING_UPDATE_AUDIT_DIR. It observes the existing optimizer and leaves
checkpointing to veRL. Receipt counters start at one for the instrumented run;
use matching receipts for loading attestation. The historical 3 GiB free-space
check and nonfinite-gradient check are retained. This hook is not installed
by default and is not executed by CPU tests.

The optional evaluation worker setup hook
`shopping_grpo.training.grpo.adapter.evaluation_audit.install` uses
`EVAL_CONDITION_DIR`, `EVAL_CHECKPOINT_STEP`, and `EVAL_TRAINING_DIR`.
It records per-rank fingerprints, requires zero LoRA B for baseline loading,
compares resumed actor weights with existing per-rank optimizer receipts and
rejects optimizer updates/checkpoint saves. These paths must refer to the
operator's own receipts. GPU checkpoint loading is not covered by CPU tests.

## Validation limits

CPU tests cover public option handling, pre-execution rejection, data-repair
scope, reviewed evidence semantics, original-score selection, collector flags,
veRL tool rejection, loss masks and paired saved-result auditing. Integration
does not establish a new model success rate. The original repository contains
failing rollout/adapter fixtures that must be reported separately from new
regressions. Run tests with the environment package on PYTHONPATH:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:.:environments/ShopSimulator/shop_env python -m pytest tests/test_enhanced_harness.py tests/test_public_support.py tests/test_budget_guard.py tests/test_task_data_quality.py tests/test_legacy_reward.py
```
