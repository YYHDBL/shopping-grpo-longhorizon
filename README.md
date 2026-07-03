# Shopping-GRPO-LongHorizon

Local baseline project for adapting ShopSimulator into an agentic GRPO-style
long-horizon shopping environment.

Current status:

- Stage 1: ShopSimulator local HTTP smoke test.
- Stage 2: semantic shopping tools, HTTP adapter, and mock rollout JSONL.
- Stage 3: veRL-style shopping context, static tool classes, and tool config.
- Stage 4: veRL-style ShopInteraction, tiny task JSONL, and dry-run configs.

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Run the Stage 2 mock rollout after starting ShopSimulator:

```bash
PYTHONPATH=src python3 scripts/run_mock_shop_rollout.py \
  --base-url http://127.0.0.1:7001 \
  --task-id 0 \
  --query '乳胶枕'
```

See:

- `docs/runbooks/stage_1_smoke_test.md`
- `docs/runbooks/stage_2_adapter.md`
- `docs/runbooks/stage_3_verl_shop_wiring.md`
- `docs/runbooks/stage_4_shop_interaction_dryrun.md`
- `docs/plans/2026-07-02-stage-2-shopsimulator-adapter.md`
