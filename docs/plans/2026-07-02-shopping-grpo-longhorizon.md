# Shopping-GRPO-LongHorizon Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the minimum baseline pipeline that connects ShopSimulator single-turn shopping tasks to agentic-grpo/veRL vanilla GRPO training.

**Architecture:** Keep ShopSimulator and agentic-grpo as upstream sibling repositories. Put project-specific adapters, configs, smoke tests, and runbooks in `shopping-grpo-longhorizon`, with the smallest wrapper layer needed to translate veRL tool calls into ShopSimulator `search[...]` / `click[...]` actions.

**Tech Stack:** Python, ShopSimulator, agentic-grpo, veRL, vLLM, Ray, PyTorch, Qwen2.5-1.5B/3B for server training.

---

## Non-Goals For Baseline

Do not implement these before the first GRPO baseline run:

```text
SFT data collection
LoRA SFT
PRM-Lite-Shopping
LATA tuning
multi-turn shopper simulator
new benchmark construction
learned reward model
large-scale training
```

Baseline means:

```text
ShopSimulator single-turn
all native action semantics adapted
environment final reward only
vanilla GRPO
group_size=2
50-100 tasks
small model policy
```

---

## Stage 0: Project Organization And References

**Purpose:** Create the project shell and collect references. No environment execution.

**Files:**
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/README.md`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/.gitignore`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/docs/references/README.md`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/docs/references/papers/ecomagentbench-2606.17698.pdf`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/docs/references/papers/shopsimulator-2601.18225.pdf`

**Validation:**

```bash
find shopping-grpo-longhorizon -maxdepth 4 -type f | sort
file shopping-grpo-longhorizon/docs/references/papers/*.pdf
```

Expected: plan/reference files exist, PDFs are valid PDF documents.

---

## Stage 1: ShopSimulator Smoke Test

**Purpose:** Verify the shopping environment works locally before touching veRL.

**Files:**
- Create: `scripts/smoke_shop_env.py`
- Create: `tests/test_shop_actions.py`

**Steps:**

1. Add a tiny client that calls the existing ShopSimulator API:

```text
POST /api/shop_agent {"action": "reset", "idx": 0}
POST /api/shop_agent {"action": "interact", "env_idx": ..., "response": "search[乳胶枕]"}
POST /api/shop_agent {"action": "release_one", "env_idx": ...}
```

2. Save one trajectory JSON under `outputs/smoke/`.

3. Verify response fields:

```text
instruction
env_idx
done
reward
reward_detail
over
```

**Validation:**

```bash
python scripts/smoke_shop_env.py --task-id 0 --actions 'search[乳胶枕]' 'click[back to search]'
```

Expected: no crash, trajectory file written, env released.

---

## Stage 2: Full Shop Action Tool Adapter

**Purpose:** Expose semantic tools while keeping ShopSimulator's native action strings hidden from the model.

**Files:**
- Create: `src/shop_actions.py`
- Create: `src/shop_client.py`
- Create: `tests/test_shop_actions.py`

**Tool Mapping:**

```text
search(query)              -> search[query]
click_product(product_id)  -> click[product_id]
select_option(value)       -> click[value]
buy_now()                  -> click[buy now]
next_page()                -> click[next >]
prev_page()                -> click[< prev]
back_to_search()           -> click[back to search]
view_description()         -> click[description]
view_features()            -> click[features]
view_reviews()             -> click[reviews]
view_attributes()          -> click[attributes]
```

**Validation:**

```bash
pytest tests/test_shop_actions.py -q
```

Expected: every semantic action converts to the exact native string above.

---

## Stage 3: ShopInteraction For veRL

**Purpose:** Mirror the TauBench adapter pattern with a ShopSimulator adapter.

**Reference Files:**

```text
../agentic-grpo-longhorizon/agentic-grpo-longhorizon/src/envs/tau_bench_interaction.py
../agentic-grpo-longhorizon/agentic-grpo-longhorizon/src/envs/tau_bench_tools.py
../agentic-grpo-longhorizon/agentic-grpo-longhorizon/src/envs/tau_bench_context.py
```

**Files:**
- Create: `src/envs/shop_context.py`
- Create: `src/envs/shop_interaction.py`
- Create: `src/envs/shop_tools.py`
- Test: `tests/test_shop_interaction.py`

**Responsibilities:**

```text
start_interaction(task_id)
reset ShopSimulator
hold env_idx and action_history
execute tool calls through ShopClient
append observation as tool result
detect done/over
return final environment reward
release env
```

**Validation:**

```bash
pytest tests/test_shop_interaction.py -q
```

Expected: fake tool-call sequence reaches done or releases env on failure.

---

## Stage 4: Local Mock Rollout

**Purpose:** Prove the project can produce trajectories without GPU training.

**Files:**
- Create: `scripts/mock_rollout.py`
- Create: `scripts/summarize_trajectories.py`

**Mock Policies:**

Use two simple policies only:

```text
fixed: search -> first product -> first option -> buy_now
random_click: search -> random clickable -> random clickable -> buy_now
```

**Validation:**

```bash
python scripts/mock_rollout.py --tasks 5 --policy fixed --out outputs/mock_rollout
python scripts/summarize_trajectories.py outputs/mock_rollout
```

Expected: summary includes `avg_reward`, `done_rate`, `buy_rate`, `invalid_action_rate`.

---

## Stage 5: veRL GRPO Baseline Config

**Purpose:** Prepare server-side training configs without running training locally.

**Files:**
- Create: `configs/tool_config/shop_tools.yaml`
- Create: `configs/interaction_config/shop_single_turn.yaml`
- Create: `configs/train/grpo/shop_vanilla_grpo.yaml`
- Create: `scripts/build_shop_grpo_data.py`
- Create: `scripts/train_shop_grpo.sh`

**Baseline Settings:**

```text
model: Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct
adv_estimator: grpo
group_size: 2
tasks: 50-100
max_turns: 10-20
reward: final ShopSimulator environment reward
rollout: vLLM on server
```

**Validation:**

```bash
python scripts/build_shop_grpo_data.py --tasks 10 --out outputs/data_check
python -m py_compile scripts/build_shop_grpo_data.py
```

Expected: train/val data files are created in the format expected by the selected veRL entrypoint.

---

## Stage 6: Evaluation Framework

**Purpose:** Evaluate a trained server checkpoint or served vLLM endpoint.

**Files:**
- Create: `configs/eval/shop_eval.yaml`
- Create: `scripts/eval_shop_policy.py`
- Create: `scripts/summarize_eval.py`

**Metrics:**

```text
avg_reward
done_rate
buy_rate
avg_turns
invalid_action_rate
reward_detail averages
```

**Validation:**

```bash
python scripts/eval_shop_policy.py --mock-policy fixed --tasks 5 --out outputs/eval_smoke
python scripts/summarize_eval.py outputs/eval_smoke
```

Expected: `eval_report.json` is created.

---

## Stage 7: Server Runbook

**Purpose:** Make server execution mechanical.

**Files:**
- Create: `docs/runbooks/server_baseline.md`

**Runbook Must Include:**

```text
environment setup
ShopSimulator service startup
vLLM policy startup
GRPO data build
GRPO training launch
checkpoint location
eval launch
common failures and fixes
```

**First Server Acceptance Target:**

```text
1-5 GRPO training steps complete
checkpoint saved
5-task eval completes
eval_report.json exists
```

---

## Stage 8: Post-Baseline Enhancements

Only after Stage 7 succeeds:

```text
SFT data collection
LoRA SFT
PRM-Lite-Shopping
LATA switch
larger task count
formal ablation
multi-turn shopper
```

---

## Baseline Completion Definition

The baseline is complete when these are true:

```text
ShopSimulator single-turn tasks run through veRL rollout
all semantic shopping tools are adapted
GRPO trains for at least one server step
checkpoint is written
eval runs against the checkpoint or vLLM endpoint
eval_report.json contains avg_reward and done_rate
```

