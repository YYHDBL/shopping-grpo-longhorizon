# Pure V4 SFT curriculum

`manifest.json` is the single training list for the active SFT recipe. It pins
the SHA256 of Pure V4, difficulty labels, and the evaluation task file; records
every train/development `task_id`; and keeps the three stages reproducible.

| Stage | Included buckets | Train | Development | Epoch | LR |
|---|---|---:|---:|---:|---:|
| A | foundation | 256 | 28 | 1 | `1e-4` |
| B | foundation + constraints | 799 | 88 | 1 | `7e-5` |
| C | all buckets | 1,073 | 119 | 1 | `5e-5` |

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
