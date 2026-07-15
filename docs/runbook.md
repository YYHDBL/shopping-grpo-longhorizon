# Runbook

Run all commands from this repository after exporting `.env` and starting ShopSimulator as described in the README.

## Task input

The collector accepts JSONL rows with either a top-level `task_id` or the older `extra_info.interaction_kwargs.task_id`. A row may also include a public `prompt`; the collector still appends the instruction returned by the environment reset. Do not put goals, answers, or reward metadata in this file.

## Resumable collection

`--attempts-per-task N` defines attempt indexes `0` through `N - 1` for every task. A raw output already containing a `(task_id, attempt_index)` pair is skipped on the next invocation. A legacy row without `attempt_index` is treated as attempt 0.

For a 6000 accepted target, start with a task list substantially larger than 6000 and a fixed attempt budget. Example: 8000 task ids with four attempts gives at most 32000 raw trajectories.

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl \
  --output outputs/accepted_6000/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" \
  --attempts-per-task 4 --temperature 0.8 --top-p 1.0 --max-steps 16

PYTHONPATH=src python3 scripts/build_sft_data.py \
  --raw outputs/accepted_6000/raw.jsonl \
  --accepted outputs/accepted_6000/accepted.jsonl \
  --rejected outputs/accepted_6000/rejected.jsonl \
  --stats outputs/accepted_6000/stats.json \
  --sft outputs/accepted_6000/sft_openai_messages.jsonl
```

If collection is interrupted, rerun the exact collector command. It appends only missing task-attempt pairs; then rerun the builder, which regenerates all four derived files from the complete raw file.

Read `outputs/accepted_6000/stats.json` after every build. Stop when `accepted >= 6000`. If all planned attempts finish below target, add more task ids or raise the attempt budget, for example from 4 to 8; existing attempts remain untouched and only indexes 4 through 7 run.

Keep a separate output directory for each sampling configuration. This makes model, temperature, task source, and retry policy auditable without adding a collector framework.
