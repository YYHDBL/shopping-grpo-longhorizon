# Stage 3 veRL Shop Wiring Runbook

Stage 3 adds veRL-style shopping tools without starting training.

Implemented:

- `src/shopping_grpo/verl_shop_context.py`
- `src/shopping_grpo/verl_shop_tools.py`
- `scripts/gen_shop_tool_config.py`
- `configs/tool_config/shop_tools.yaml`

## Regenerate Tool Config

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 scripts/gen_shop_tool_config.py \
  --output configs/tool_config/shop_tools.yaml
```

The output file is JSON-formatted YAML so local validation can use stdlib only:

```bash
python3 -m json.tool configs/tool_config/shop_tools.yaml >/dev/null
```

## Run Tests

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

## What This Proves

- A shopping tool class can be instantiated with a veRL-style schema.
- `execute()` reads the per-trajectory env/state from contextvars.
- `search_products` maps to `search[...]` and calls `env.step`.
- `think` records state without touching the environment.
- `configs/tool_config/shop_tools.yaml` points to static tool classes.

## Stage 4 Target

Stage 4 should add the full rollout integration:

- a veRL `Interaction` class that creates `ShopHttpEnv`;
- prompt parquet or a tiny JSONL-to-parquet builder;
- `configs/interaction_config/shop.yaml`;
- `configs/train/grpo/shop_tiny_grpo.yaml`;
- one server-side dry-run using small model and `group_size=2`.

ponytail: keep Stage 4 search-only until one veRL rollout batch finishes.
