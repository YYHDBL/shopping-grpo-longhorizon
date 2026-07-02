# Stage 2 Adapter Runbook

Stage 2 wires a minimal shopping adapter:

```text
tool call -> search/click action -> ShopSimulator HTTP app -> JSONL trajectory
```

It does not start vLLM or GRPO.

## Start ShopSimulator

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env
PYTHONPATH=. SHOPSIM_ALLOW_LINEAR_SEARCH=1 SHOPSIM_APP_PORT=7001 .venv/bin/python web_agent_site/app.py
```

Expected listener:

```text
Running on http://127.0.0.1:7001
```

## Run Unit Tests

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

## Run Mock Rollout

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 scripts/run_mock_shop_rollout.py \
  --base-url http://127.0.0.1:7001 \
  --task-id 0 \
  --query '乳胶枕'
```

Default output:

```bash
outputs/rollouts/stage2_mock.jsonl
```

## Validate JSONL

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("outputs/rollouts/stage2_mock.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert rows
assert rows[0]["steps"]
print(path, len(rows), rows[0]["steps"][0]["info"]["status_code"])
PY
```

Expected status code is `200`.

## Implemented Files

- `src/shopping_grpo/shop_http_env.py`: tiny HTTP wrapper around ShopSimulator.
- `src/shopping_grpo/shop_tools.py`: semantic shopping tools mapped to `search[...]` and `click[...]`.
- `src/shopping_grpo/rollout.py`: minimal trajectory and JSONL helpers.
- `scripts/run_mock_shop_rollout.py`: search-only mock policy runner.

## Stage 3 Handoff

Stage 3 should add veRL-specific wiring:

- `src/shopping_grpo/verl_shop_context.py`
- `src/shopping_grpo/verl_shop_tools.py`
- `configs/tool_config/shop_tools.yaml`
- `configs/interaction_config/shop.yaml`
- `configs/train/grpo/shop_mock_grpo.yaml`

ponytail: keep Stage 3 search-only first; add product selection after the GRPO dry-run can consume one stable trajectory.
