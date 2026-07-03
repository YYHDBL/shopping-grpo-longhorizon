# Stage 4 Shop Interaction Dry-Run Runbook

Stage 4 adds the veRL Interaction side of the shopping pipeline.

Implemented:

- `src/shopping_grpo/verl_shop_interaction.py`
- `configs/interaction_config/shop.yaml`
- `configs/train/grpo/shop_tiny_grpo.yaml`
- `data/shop_tiny_tasks.jsonl`
- `scripts/validate_shop_tiny_dataset.py`
- `scripts/dryrun_shop_interaction.py`

## Local Validation

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m json.tool configs/interaction_config/shop.yaml >/dev/null
python3 -m json.tool configs/train/grpo/shop_tiny_grpo.yaml >/dev/null
PYTHONPATH=src python3 scripts/validate_shop_tiny_dataset.py data/shop_tiny_tasks.jsonl
PYTHONPATH=src python3 scripts/dryrun_shop_interaction.py --fake-env
```

## Optional HTTP Dry-Run

Start ShopSimulator first:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env
PYTHONPATH=. SHOPSIM_ALLOW_LINEAR_SEARCH=1 SHOPSIM_APP_PORT=7001 .venv/bin/python web_agent_site/app.py
```

Then run:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
PYTHONPATH=src python3 scripts/dryrun_shop_interaction.py --query '乳胶枕'
```

## Server-Side Parquet Target

Convert `data/shop_tiny_tasks.jsonl` into:

```text
experiments/shop_tiny/train.parquet
experiments/shop_tiny/val.parquet
```

Required row shape:

```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "extra_info": {
    "index": 0,
    "task_id": 0,
    "split": "train",
    "interaction_kwargs": {"name": "shop", "task_id": 0}
  }
}
```

## GRPO Dry-Run Target

Use the config skeleton:

```bash
configs/train/grpo/shop_tiny_grpo.yaml
```

The first server objective is only:

```text
one batch starts -> model sees shop tools -> search tool executes -> reward path returns -> trainer exits after a few steps
```

ponytail: do not add PRM-Lite or product-selection reward until this dry-run completes.
