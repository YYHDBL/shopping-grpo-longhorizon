# 数据索引：什么在 git 里，什么在本地产

> 更新：2026-10-08。仓库只携带复现所需的轻量数据（~15M）；大数据留在本地，列在此处备查。

## 已入库（clone 即有）

| 路径 | 大小 | 内容 |
|---|---|---|
| `outputs/split/tasks_final.jsonl` | 3.8M | **19,106 条已验收任务的定稿**（split/difficulty/persona 引用/tag）——一切数据的源头 |
| `outputs/persona/persona_pool.jsonl` | 8.1M | 画像池 4,666 份（tasks_final 通过 persona_ref 引用） |
| `outputs/acceptance/final_verdicts.jsonl` | 2.2M | 数据验收定案（23,421 → 19,106 的逐条判定） |
| `outputs/grpo/train_1000.parquet` | 314K | run1/run2 的 RL 任务集（1000 题，∩eval=0） |
| `outputs/grpo/train_1000_curriculum.parquet` | 316K | run3/run4 的课程制重排版（任务集相同，仅顺序） |
| `outputs/grpo/smoke.parquet` | 31K | 64 题训练中验证集 |
| `outputs/trace_phase0/targets.jsonl` | 112K | 1092 个评测任务的 gold target（TRACE 验证用） |
| `outputs/trace_phase0/credits_run3_50.jsonl` | 157K | 163 条轨迹的 turn-level credit（TRACE Phase 0） |
| `data/grpo/` | ~200K | 早期数据集（**已弃用**，隔离性违规，仅存档；背景见 work-summary） |

## 未入库（本地大文件）

| 路径 | 大小 | 说明 / 重新生成方式 |
|---|---|---|
| `outputs/collection/trajectories/` | 1.1G | Teacher 采集原始轨迹（教材来源）——重采集需双 Teacher API |
| `outputs/evaluation/*/trajectories.jsonl` | 872M | 六臂 × 1092 题评测轨迹——重跑评测约 1 GPU 时/臂 |
| `outputs/sft_dataset/` | 119M | SFT 训练 parquet（5,542+110）——由教材 + tasks_final 加工生成 |
| `outputs/models/*`、`checkpoints/*` | ~90G | 模型权重（sft-merged / grpo 各 checkpoint）——训练产物 |
| `outputs/acceptance/jev_verdicts.jsonl` 等 | 25M | 验收中间数据（final_verdicts 已定案） |

## 复现最短路径（有 GPU 环境时）

```text
tasks_final.jsonl（已入库）→ 环境回放 + 采集（或直接使用采集产物）
→ SFT 数据集加工 → SFT 训练（sft-merged）
→ GRPO：outputs/grpo/train_1000_curriculum.parquet + 配置 configs/grpo_run3.yaml
→ 评测：冻结 1092 题（tasks_final 的 eval split）+ scripts/evaluate_model.py
```

实验的完整决策记录与结果见 `docs/plans/2026-09-27-work-summary.md`。
