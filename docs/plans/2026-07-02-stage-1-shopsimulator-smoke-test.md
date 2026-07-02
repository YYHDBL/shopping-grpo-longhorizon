# Stage 1 Plan: ShopSimulator Smoke Test

Status: completed for local baseline smoke.

## Goal

Prove that the local project can call ShopSimulator through an HTTP entrypoint
and save a minimal trajectory artifact before starting GRPO integration.

## Completed

- Created a `uv` Python 3.10 virtual environment under ShopSimulator.
- Installed the minimal dependency set needed for local smoke.
- Generated `data/items_eval_train.json` from the bundled gzipped data.
- Added configurable ports:
  - `SHOPSIM_PORT` for `shop_env/pack_api.py`
  - `SHOPSIM_APP_PORT` for `web_agent_site/app.py`
- Added `SHOPSIM_ALLOW_LINEAR_SEARCH=1` fallback because the local clone lacks
  `shop_env/search_engine/indexes`.
- Rebuilt the project smoke script and tests.
- Verified HTTP smoke against `http://127.0.0.1:7001`.

## Evidence

Commands verified:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
python3 -m unittest discover -s tests -p 'test_*.py'
```

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_linear_search_fallback.py'
```

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
python3 scripts/smoke_shop_env.py --base-url http://127.0.0.1:7001 --task-id 0 --actions 'search[乳胶枕]'
```

Latest smoke artifact:

```bash
/Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon/outputs/smoke/task_0000_20260702_230627.json
```

## Stage 2 Gate

Stage 2 can start now. The next stage should build a proper adapter layer:

- define the action schema for ShopSimulator tools;
- expose a single-step rollout API usable by GRPO;
- convert model text/tool calls into ShopSimulator HTTP/environment actions;
- save trajectories in the format expected by the training framework.
