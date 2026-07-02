# Stage 2 ShopSimulator Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the minimal ShopSimulator adapter that converts model/tool calls into shopping environment actions and records rollout trajectories.

**Architecture:** Keep Stage 2 as a thin adapter layer. The adapter talks to the already-verified ShopSimulator Flask app, exposes semantic shopping tools, validates click actions against the current page when possible, and writes JSONL trajectories that Stage 3 can feed into GRPO/veRL integration.

**Tech Stack:** Python stdlib, local ShopSimulator HTTP app, existing `shopping-grpo-longhorizon` scripts/tests, later-compatible veRL-style tool schema.

---

## Scope

Stage 2 does:

- Create a local Python package under `shopping-grpo-longhorizon/src/shopping_grpo`.
- Implement one HTTP environment wrapper around ShopSimulator.
- Implement semantic shopping tools backed by `search[...]` and `click[...]`.
- Add a mock rollout runner that uses a hand-written policy, not a model.
- Save trajectories as JSONL for later GRPO wiring.

Stage 2 does not:

- Start vLLM.
- Train GRPO.
- Implement PRM-Lite shopping rules.
- Use `pack_api.py` multi-env service.
- Solve missing Lucene index beyond the Stage 1 linear-search fallback.

## Current Facts

- Stage 1 smoke verified `http://127.0.0.1:7001`.
- ShopSimulator valid high-level actions are `search[...]` and `click[...]`.
- `web_agent_site/envs/web_agent_site_env.py` documents:
  - `search[keywords]`
  - `click[value]`
- `agentic-grpo-longhorizon` tool pattern is:
  - Interaction creates/binds env state.
  - Tool class reads env from context.
  - Tool `execute()` calls env step.
  - Tool returns observation text and zero step reward.
- For Stage 2, we mimic the shape without importing veRL yet.

## Tool Set

Use separate semantic tools, but map them to the same simple backend action:

| Tool | Parameters | Backend action |
| --- | --- | --- |
| `search_products` | `query: str` | `search[query]` |
| `open_product` | `asin: str` | `click[asin]` |
| `select_option` | `value: str` | `click[value]` |
| `view_description` | none | `click[Description]` |
| `view_features` | none | `click[Features]` |
| `view_reviews` | none | `click[Reviews]` |
| `view_attributes` | none | `click[Attributes]` |
| `next_page` | none | `click[Next >]` |
| `prev_page` | none | `click[< Prev]` |
| `back_to_search` | none | `click[Back to Search]` |
| `buy_now` | none | `click[Buy Now]` |
| `think` | `note: str` | no-op, records reasoning only |

ponytail: these are many schemas but one implementation path; split tools help the model, shared backend keeps code small.

## Done Definition

Stage 2 is done when:

- Unit tests pass.
- A local mock rollout runs against the Flask app.
- At least one JSONL trajectory contains:
  - task id
  - instruction text if parseable
  - action list
  - observations
  - final reward
  - done flag
- No Flask server is left running by tests.

## Task 1: Package Skeleton

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/__init__.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_imports.py`

**Step 1: Write failing test**

```python
def test_package_imports():
    import shopping_grpo
    assert shopping_grpo.__version__
```

**Step 2: Run test**

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: fail because package does not exist.

**Step 3: Implement minimum package**

```python
__version__ = "0.1.0"
```

**Step 4: Verify**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: OK.

## Task 2: HTTP Environment Wrapper

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/shop_http_env.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_shop_http_env.py`

**Step 1: Write tests for URL building**

```python
from shopping_grpo.shop_http_env import ShopHttpEnv


def test_build_search_url():
    env = ShopHttpEnv(base_url="http://127.0.0.1:7001")
    env.session_id = "fixed_0"
    assert "%E4%B9%B3%E8%83%B6%E6%9E%95" in env.build_action_url("search[乳胶枕]")


def test_build_click_url_rejects_without_page_state():
    env = ShopHttpEnv(base_url="http://127.0.0.1:7001")
    env.session_id = "fixed_0"
    assert env.build_action_url("click[Buy Now]") is None
```

**Step 2: Run and confirm fail**

```bash
PYTHONPATH=src python3 -m unittest tests/test_shop_http_env.py
```

Expected: fail because `ShopHttpEnv` is missing.

**Step 3: Implement minimum wrapper**

Implement:

- `reset(task_id: int) -> dict`
- `step(action: str) -> dict`
- `build_action_url(action: str) -> str | None`
- `parse_observation(html: str) -> dict`

Use stdlib only:

- `urllib.request` for GET.
- `html.parser.HTMLParser` for rough text/action extraction.

Return shape:

```python
{
    "observation": "...text...",
    "html": "...",
    "url": "...",
    "reward": 0.0,
    "done": False,
    "available_actions": ["..."],
}
```

ponytail: rough HTML parsing is enough for Stage 2; replace with BeautifulSoup only if stdlib parsing breaks on real pages.

**Step 4: Verify**

```bash
PYTHONPATH=src python3 -m unittest tests/test_shop_http_env.py
```

Expected: OK.

## Task 3: Semantic Tool Layer

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/shop_tools.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_shop_tools.py`

**Step 1: Write tests for tool mapping**

```python
from shopping_grpo.shop_tools import tool_call_to_action


def test_search_products_maps_to_search_action():
    assert tool_call_to_action("search_products", {"query": "乳胶枕"}) == "search[乳胶枕]"


def test_buy_now_maps_to_click_action():
    assert tool_call_to_action("buy_now", {}) == "click[Buy Now]"
```

**Step 2: Run and confirm fail**

```bash
PYTHONPATH=src python3 -m unittest tests/test_shop_tools.py
```

Expected: fail because module is missing.

**Step 3: Implement minimum mapping**

Implement one function:

```python
def tool_call_to_action(name: str, parameters: dict) -> str:
    ...
```

Also expose `SHOP_TOOL_SCHEMAS`, a plain list of OpenAI-style function schemas.

**Step 4: Verify**

```bash
PYTHONPATH=src python3 -m unittest tests/test_shop_tools.py
```

Expected: OK.

## Task 4: Rollout State And Trajectory Format

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/src/shopping_grpo/rollout.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_rollout.py`

**Step 1: Write failing test**

```python
from shopping_grpo.rollout import make_trajectory


def test_make_trajectory_has_required_fields():
    traj = make_trajectory(task_id=0, steps=[], final_reward=0.0, done=False)
    assert set(["task_id", "steps", "final_reward", "done"]).issubset(traj)
```

**Step 2: Run and confirm fail**

```bash
PYTHONPATH=src python3 -m unittest tests/test_rollout.py
```

Expected: fail because module is missing.

**Step 3: Implement minimum**

Implement:

- `make_step(tool_name, parameters, action, observation, reward, done, info)`
- `make_trajectory(task_id, steps, final_reward, done)`
- `write_jsonl(path, trajectories)`

**Step 4: Verify**

```bash
PYTHONPATH=src python3 -m unittest tests/test_rollout.py
```

Expected: OK.

## Task 5: Mock Policy Runner

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/scripts/run_mock_shop_rollout.py`
- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/tests/test_mock_rollout.py`

**Step 1: Write test for policy output shape**

```python
from scripts.run_mock_shop_rollout import build_mock_actions


def test_mock_actions_are_tool_calls():
    actions = build_mock_actions(query="乳胶枕")
    assert actions[0]["name"] == "search_products"
    assert "parameters" in actions[0]
```

**Step 2: Run and confirm fail**

```bash
PYTHONPATH=src python3 -m unittest tests/test_mock_rollout.py
```

Expected: fail because script is missing.

**Step 3: Implement CLI**

Arguments:

- `--base-url`, default `http://127.0.0.1:7001`
- `--task-id`, default `0`
- `--query`, default `乳胶枕`
- `--output`, default `outputs/rollouts/stage2_mock.jsonl`

Mock actions:

```python
[
    {"name": "search_products", "parameters": {"query": query}},
]
```

Do not attempt product selection in the first Stage 2 runner. Search-only is enough to prove adapter wiring.

**Step 4: Verify with running Flask app**

Start ShopSimulator:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env
PYTHONPATH=. SHOPSIM_ALLOW_LINEAR_SEARCH=1 SHOPSIM_APP_PORT=7001 .venv/bin/python web_agent_site/app.py
```

Run:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 scripts/run_mock_shop_rollout.py --base-url http://127.0.0.1:7001 --task-id 0 --query '乳胶枕'
```

Expected:

- Output JSONL exists.
- One trajectory is written.
- It has at least one step.
- Step status is HTTP 200.

## Task 6: veRL Compatibility Notes

**Files:**

- Create: `/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/docs/runbooks/stage_2_adapter.md`

**Step 1: Document local commands**

Include:

- How to start ShopSimulator Flask app.
- How to run tests.
- How to run mock rollout.
- Where JSONL is saved.

**Step 2: Document Stage 3 handoff**

Stage 3 should add veRL-specific files:

- `src/shopping_grpo/verl_shop_context.py`
- `src/shopping_grpo/verl_shop_tools.py`
- `configs/tool_config/shop_tools.yaml`
- `configs/interaction_config/shop.yaml`
- `configs/train/grpo/shop_mock_grpo.yaml`

Do not create these in Stage 2 unless the adapter smoke is already stable.

## Verification Commands

Run all Stage 2 unit checks:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Run local HTTP smoke:

```bash
PYTHONPATH=src python3 scripts/run_mock_shop_rollout.py --base-url http://127.0.0.1:7001 --task-id 0 --query '乳胶枕'
```

Validate JSONL:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("outputs/rollouts/stage2_mock.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert rows
assert rows[0]["steps"]
print(path, len(rows), rows[0]["final_reward"], rows[0]["done"])
PY
```

## Stage 3 Gate

Only start Stage 3 after Stage 2 produces stable JSONL rollouts. Stage 3 is the first place to wire this into veRL/GRPO:

- static veRL tool classes;
- Hydra tool config;
- interaction config;
- prompt dataset;
- tiny GRPO dry-run config.

