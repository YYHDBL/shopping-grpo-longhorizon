# Stage 4 Shop Interaction Dry-Run Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal veRL-compatible shopping Interaction and dry-run assets for the next server-side GRPO test.

**Architecture:** Keep the proven Stage 2/3 adapter and tool classes. Add one `ShopInteraction` that creates `ShopHttpEnv`, binds contextvars, exposes veRL `start_interaction/generate_response/calculate_score/finalize_interaction`, and add config/dataset files that point veRL to the shopping tools.

**Tech Stack:** Python stdlib, unittest, JSON-as-YAML configs, existing `shopping_grpo` package.

---

## Scope

Stage 4 does:

- Add `src/shopping_grpo/verl_shop_interaction.py`.
- Add `configs/interaction_config/shop.yaml`.
- Add `configs/train/grpo/shop_tiny_grpo.yaml`.
- Add tiny dataset JSONL and a validator.
- Add a local dry-run script that exercises `ShopInteraction + ShopTool`.

Stage 4 does not:

- Start vLLM.
- Train GRPO.
- Generate parquet locally.
- Implement PRM-Lite shopping reward.

ponytail: JSONL first; parquet conversion belongs on the server where pandas/pyarrow already exist.

## Task 1: ShopInteraction

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/verl_shop_interaction.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_verl_shop_interaction.py`

**Steps:**

1. Write failing tests for `start_interaction`, context binding, `calculate_score`, and `finalize_interaction`.
2. Implement a local fallback `BaseInteraction` if veRL is not installed.
3. Use `ShopHttpEnv` by default, with `env_factory` only for tests.
4. Run all tests.

## Task 2: Config Files

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/configs/interaction_config/shop.yaml`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/configs/train/grpo/shop_tiny_grpo.yaml`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_stage4_configs.py`

**Steps:**

1. Write failing tests that JSON-load both config files.
2. Add JSON-formatted YAML files.
3. Verify tool config path, interaction name, and GRPO group size `n=2`.
4. Run all tests.

## Task 3: Tiny Dataset JSONL

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/data/shop_tiny_tasks.jsonl`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/scripts/validate_shop_tiny_dataset.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_shop_tiny_dataset.py`

**Steps:**

1. Write failing test for two dataset rows with `extra_info.interaction_kwargs.name == "shop"`.
2. Add 4 tiny tasks.
3. Add validator script using stdlib only.
4. Run all tests.

## Task 4: Local Dry-Run

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/scripts/dryrun_shop_interaction.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_dryrun_shop_interaction.py`

**Steps:**

1. Write failing test for a fake-env dry-run function returning one tool call.
2. Implement dry-run using `ShopInteraction` and `Shop_search_products_Tool`.
3. Keep HTTP mode optional via CLI.
4. Run all tests.

## Task 5: Runbook

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/docs/runbooks/stage_4_shop_interaction_dryrun.md`
- Update: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/README.md`

**Steps:**

1. Document local validation.
2. Document server-side parquet conversion target.
3. Document server-side GRPO dry-run command skeleton.

## Verification

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m json.tool configs/interaction_config/shop.yaml >/dev/null
python3 -m json.tool configs/train/grpo/shop_tiny_grpo.yaml >/dev/null
PYTHONPATH=src python3 scripts/validate_shop_tiny_dataset.py data/shop_tiny_tasks.jsonl
PYTHONPATH=src python3 scripts/dryrun_shop_interaction.py --fake-env
```

