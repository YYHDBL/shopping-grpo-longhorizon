# Stage 3 veRL Shop Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the minimal veRL-style shopping tool wiring needed for a later GRPO rollout dry-run.

**Architecture:** Reuse Stage 2 `ShopHttpEnv` and semantic tools. Add contextvars for per-trajectory env/state, static tool classes with the same shape as agentic-grpo tau-bench tools, and a generated tool config that veRL can load when `PYTHONPATH=src`.

**Tech Stack:** Python stdlib, existing `shopping_grpo` package, JSON-as-YAML configs, unittest.

---

## Scope

Stage 3 does:

- Add `verl_shop_context.py`.
- Add `verl_shop_tools.py`.
- Generate `configs/tool_config/shop_tools.yaml`.
- Add tests proving the veRL-style tool class can execute against a fake env.
- Add a runbook for the next GRPO dry-run step.

Stage 3 does not:

- Start vLLM.
- Start GRPO training.
- Implement a full veRL `Interaction` class.
- Add learned reward or PRM-Lite.

ponytail: full Interaction wiring waits until the tool class and config are stable; otherwise we debug two integration surfaces at once.

## Task 1: Context

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/verl_shop_context.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_verl_shop_context.py`

**Steps:**

1. Write a failing test for `make_initial_state(task_id=3)`.
2. Run `PYTHONPATH=src python3 -m unittest tests/test_verl_shop_context.py`.
3. Implement contextvars and `make_initial_state`.
4. Run all tests.

## Task 2: veRL-style Tool Classes

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/verl_shop_tools.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_verl_shop_tools.py`

**Steps:**

1. Write a failing async test for `Shop_search_products_Tool.execute`.
2. Use a fake env with `step(action)` returning a dict.
3. Implement a fallback `BaseTool`/`ToolResponse` shim for local tests when veRL is not installed.
4. Implement `ShopToolBase.execute`.
5. Add static classes for every Stage 2 tool.
6. Run all tests.

## Task 3: Tool Config Generator

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/scripts/gen_shop_tool_config.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/configs/tool_config/shop_tools.yaml`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_shop_tool_config.py`

**Steps:**

1. Write a failing test that loads `shop_tools.yaml` as JSON and checks `search_products`.
2. Implement the generator using stdlib `json`.
3. Generate the config.
4. Run all tests.

## Task 4: Runbook

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/docs/runbooks/stage_3_verl_shop_wiring.md`

**Steps:**

1. Document tests.
2. Document config generation.
3. Document the next Stage 4 target: full veRL Interaction + tiny GRPO dry-run.

## Verification

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python3 scripts/gen_shop_tool_config.py --output configs/tool_config/shop_tools.yaml
python3 -m json.tool configs/tool_config/shop_tools.yaml >/dev/null
```

