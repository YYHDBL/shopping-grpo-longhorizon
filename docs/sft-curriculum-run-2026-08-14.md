# 三阶段 SFT 课程训练报告

> 运行日期：2026-08-14。本文只记录可复现的配置、摘要指标和产物位置；模型、checkpoint、轨迹、缓存和评测 HTML 均保留在本地数据盘，不纳入 Git。

## 运行范围

- 基础模型：`Qwen/Qwen3.5-2B`
- 环境：ShopSimulator v2.1；Reward v3；observation v2；tool schema v2
- 精度与注意力：BF16、SDPA、gradient checkpointing、LoRA
- 课程顺序：Stage A → Stage B → Stage C
- SwanLab 项目：`shopping-grpo-sft-curriculum`
- 训练数据清单：`data/sft_curriculum/manifest.json`
- 课程清单统计：总计 1192；训练 1073；开发集 119；`evaluation_overlap=0`
- 课程目标训练量：A=256，B=799，C=1073

## SFT 结果

| 阶段 | 训练样本 | 开发集样本 | 训练耗时 | train_loss | eval_loss | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| A | 256 | 28 | 15.0 min | 0.4220 | 0.3537 | 35.79 GiB |
| B | 799 | 88 | 52.0 min | 0.3595 | 0.3386 | 49.93 GiB |
| C | 1069* | 118* | 78.1 min | 0.3321 | 0.3480 | 68.97 GiB |

\* manifest 中 C 的目标训练量和开发集分别为 1073 和 119；运行时 tokenization 记录为保留 1069/118、丢弃 1 条。未修改训练数据或课程配置，该差异作为异常保留。

每个阶段均已生成 adapter、最终 checkpoint、`train_summary.json` 和 merged 模型。最终 GRPO 起始模型为：

```text
outputs/models/sft-curriculum/stage-c/merged
```

## Final-200 Clean 结果

三次评测均使用 `data/evaluation/tasks.jsonl`、Reward v3 和统一评测协议；没有使用 Final-200 结果选择 checkpoint。

| 模型 | 完成任务 | 严格成功 / gold_purchase | reward_valid | 平均最终奖励 | 平均步数 |
|---|---:|---:|---:|---:|---:|
| Stage A | 200/200 | 111/200 (55.5%) | 98% | 0.3785 | 7.33 |
| Stage B | 200/200 | 118/200 (59.0%) | 99% | 0.4177 | 8.115 |
| Stage C | 200/200 | 122/200 (61.0%) | 97% | 0.4685 | 8.27 |

Stage C 的单任务状态为：`done=196`、`assistant_final=1`、`error=1`、`invalid_action_limit=2`。唯一基础设施之外的 error 是 task `12353` 的 `ContextBudgetError`；评测仍完成全部 200 个任务并生成 summary。第一次因 vLLM 尚未就绪导致的评测失败单独保留，未与本次重跑结果混合。

## 线上记录

- Stage A：<https://swanlab.cn/@yyhdbl/shopping-grpo-sft-curriculum/runs/8qz181o3>
- Stage B：<https://swanlab.cn/@yyhdbl/shopping-grpo-sft-curriculum/runs/18iesiog>
- Stage C：<https://swanlab.cn/@yyhdbl/shopping-grpo-sft-curriculum/runs/hzk05pf0>

本地完整产物位于数据盘：

```text
/root/autodl-tmp/shopping-grpo-longhorizon/outputs/models/sft-curriculum/
/root/autodl-tmp/shopping-grpo-longhorizon/outputs/evaluation/
```

## 下一阶段：GRPO

三阶段 SFT 和 A/B/C 对比评测已完成，下一阶段是 GRPO。启动前仍需确认 GRPO 使用的配置和 rollout 预算；默认起始模型应为：

```text
outputs/models/sft-curriculum/stage-c/merged
```

本次报告提交不启动 GRPO。
