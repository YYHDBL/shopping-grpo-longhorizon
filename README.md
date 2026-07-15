# Shopping-GRPO-LongHorizon

Local baseline project for adapting ShopSimulator into an agentic GRPO-style
long-horizon shopping environment.

Current status:

- Stage 1: ShopSimulator local HTTP smoke test.
- Stage 2: OpenAI-compatible teacher rollout collector and raw trajectory JSONL.
- Stage 3: veRL-style shopping context, static tool classes, and tool config.
- Stage 4: veRL-style ShopInteraction, tiny task JSONL, and dry-run configs.

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Run the Stage 2 mock rollout after starting ShopSimulator:

```bash
PYTHONPATH=src python3 scripts/run_mock_shop_rollout.py \
  --base-url http://127.0.0.1:5000 \
  --task-id 0 \
  --query '乳胶枕'
```

Collect Stage 2 teacher rollouts after starting ShopSimulator:

```bash
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export OPENAI_API_KEY=...
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --base-url http://127.0.0.1:5000 \
  --tasks data/shop_tiny_tasks.jsonl \
  --output outputs/rollouts/teacher_raw.jsonl \
  --model deepseek-chat \
  --max-steps 8
```

See:

- `docs/runbooks/stage_1_smoke_test.md`
- `docs/runbooks/stage_2_adapter.md`
- `docs/runbooks/stage_3_verl_shop_wiring.md`
- `docs/runbooks/stage_4_shop_interaction_dryrun.md`
- `docs/plans/2026-07-02-stage-2-shopsimulator-adapter.md`
