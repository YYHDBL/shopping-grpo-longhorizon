# Data

V2 活动数据只保留 GRPO 提示集与环境冻结清单：

| Stage | Files | Rows |
|---|---|---:|
| GRPO | `grpo/train.parquet`, `grpo/validation.parquet` | 1000 / 50 |
| Environment | `environment.json`（Environment v2.1 冻结 manifest） | — |

`grpo/metadata.json` 记录 SHA256 校验与来源。SFT 教材由
`scripts/build_sft_dataset.py` + `scripts/export_sft_parquet.py` 生成在
`outputs/sft_dataset/`，评测任务来自 `outputs/split/tasks_final.jsonl` 的
冻结 `tag=eval` 切分；生成产物一律落在 `outputs/`，不进入 `data/`。
V1 时代的 Final-200 评测、SFT 教材与课程数据已随 V2 清理退出，仅存于
Git 历史。
