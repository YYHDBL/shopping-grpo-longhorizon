# Pure V4 SFT curriculum

`manifest.json` is the single training list for the active SFT recipe. It pins
the SHA256 of Pure V4, difficulty labels, and the evaluation task file; records
every train/development `task_id`; and keeps the three stages reproducible.

| Stage | Included buckets | Train | Development | Epoch | LR |
|---|---|---:|---:|---:|---:|
| A | foundation | 256 | 28 | 1 | `1e-4` |
| B | foundation + constraints | 799 | 88 | 1 | `7e-5` |
| C | all buckets | 1,073 | 119 | 1 | `5e-5` |

## Curriculum weighting audit

The Pure V4 file itself does not contain a 100x duplicated task: it has 1,192
rows and 1,192 unique task IDs. The merge audit removed 258 duplicate rows from
258 duplicate task groups, and exact duplicate user prompts were not found in
the resulting file. See `../sft_pure_v4/metadata.json` and
`../sft_pure_v4/duplicate_report.json`.

The curriculum nevertheless creates implicit reweighting because each stage
starts from the previous merged checkpoint and trains for one epoch:

| Bucket | Unique train rows | Stages | Effective exposures per task |
|---|---:|---|---:|
| foundation (all simple) | 256 | A, B, C | 3 |
| constraints (all medium) | 543 | B, C | 2 |
| strategy (174 medium, 100 hard) | 274 | C | 1 |

This produces 2,128 effective row exposures rather than 1,073. The raw train
mix is 23.9% simple / 66.8% medium / 9.3% hard; after cumulative exposure it is
approximately 36.1% / 59.2% / 4.7%. With the stage learning rates, the nominal
per-task LR exposure is `2.2e-4` for foundation, `1.2e-4` for constraints, and
`5e-5` for strategy. This is not an exact optimizer weight because Adam state
and changing checkpoints matter, but it is a real bias toward foundation and
away from hard strategy tasks.

This is a plausible reason for weak curriculum gains: the later stage adds the
hardest examples only once, while repeatedly revisiting easier examples. It is
a hypothesis, not yet a causal result; compare against a single-pass baseline
or log per-task exposure counts before changing the curriculum. Do not fix this
by blindly duplicating hard rows, since that would introduce the same synthetic
reweighting problem in another form.

Regenerate and audit the list after changing either source file:

```bash
.venv/bin/python scripts/prepare_sft_curriculum.py
git diff -- data/sft_curriculum/manifest.json
```

Server use:

```bash
bash scripts/setup.sh
bash scripts/sft_curriculum.sh --dry-run
bash scripts/sft_curriculum.sh --swanlab
```

如服务器使用外部虚拟环境，可设置
`SFT_PYTHON=/path/to/venv/bin/python`，不需要改脚本。

To continue after a completed stage:

```bash
bash scripts/sft_curriculum.sh --start-stage b --swanlab
```

To resume an interrupted stage, point at its Transformers checkpoint:

```bash
bash scripts/sft_curriculum.sh \
  --start-stage b \
  --resume-from-checkpoint outputs/models/sft-curriculum/stage-b/adapter/checkpoint-100 \
  --swanlab
```

`review_flags` are triage lists, not automatic deletions. Search-heavy or long
trajectories can be useful hard examples. Final-200 Clean is excluded from all
gradient rows and must not be used to tune these stages. The GRPO base is:

```text
outputs/models/sft-curriculum/stage-c/merged
```
