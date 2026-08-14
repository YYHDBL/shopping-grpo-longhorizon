# SFT → GRPO 实验记录

运行日期：2026-08-14

本文记录本次从三阶段 SFT 到 GRPO 训练、checkpoint 导出和 Final-200
Clean 评测的可复核结果。模型权重、checkpoint、完整 rollout、缓存和 HTML
报告均保留在本地数据盘的 `outputs/`，不上传到 GitHub。

## 1. 固定运行协议

- 基础模型：`Qwen/Qwen3.5-2B`
- ShopSimulator：Environment v2.1
- Reward：v3
- Observation：v2
- Tool schema：v2
- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，约 96 GiB
- SFT 精度：BF16、SDPA、LoRA、gradient checkpointing
- Final-200 Clean：`data/evaluation/tasks.jsonl`，200 个留出任务
- SFT、GRPO 训练数据与 Final-200 的 task ID overlap：0

所有正式评测都使用确定性推理：temperature `0.0`、max steps `35`、max
tokens `512`、context window `24576`。

## 2. SFT 课程训练

SFT 使用固定课程清单 `data/sft_curriculum/manifest.json`。三阶段是累计课程，
不是三个互不重叠的数据集：

| 阶段 | 训练桶 | 训练目标量 | 开发集 | 学习率 | 训练耗时 |
|---|---|---:|---:|---:|---:|
| Stage A | foundation | 256 | 28 | `1e-4` | 约 15.0 min |
| Stage B | foundation + constraints | 799 | 88 | `7e-5` | 约 52.0 min |
| Stage C | foundation + constraints + strategy | 1,073 | 119 | `5e-5` | 约 78.1 min |

全量训练集的难度配比为：simple 284（23.8%）、medium 798（66.9%）、hard
110（9.2%）。课程中的重复暴露关系是：simple 样本经过 A/B/C 三次，基础
medium 样本经过 B/C 两次，strategy/hard 样本在 C 阶段加入。

Stage C 运行时 tokenization 统计为 1,069 条训练样本和 118 条开发样本，
manifest 目标仍是 1,073/119；该差异已保留为运行异常记录，没有修改数据或课程清单。

### SFT 训练与 Final-200 结果

| 阶段 | train loss | eval loss | 峰值显存 | 严格成功 | 平均 reward | 完成率 | 平均步数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage A | 0.4220 | 0.3537 | 35.79 GiB | 111/200（55.5%） | 0.3785 | 100.0% | 7.33 |
| Stage B | 0.3595 | 0.3386 | 49.93 GiB | 118/200（59.0%） | 0.4177 | 99.0% | 8.115 |
| Stage C | 0.3321 | 0.3480 | 68.97 GiB | 122/200（61.0%） | 0.4685 | 98.0% | 8.27 |

每个阶段都生成了 adapter、checkpoint、`train_summary.json` 和 merged 模型。
GRPO 的起始模型是：

```text
outputs/models/sft-curriculum/stage-c/merged
```

## 3. GRPO 训练

### 第一次尝试

第一次使用仓库现有 `scripts/grpo.sh` 和默认 `configs/grpo.yaml`，训练到
optimizer step 11 时发生 CUDA OOM：

```text
torch.OutOfMemoryError: Tried to allocate 17.73 GiB
```

当时显存约 85.74 GiB 已被训练进程使用，错误发生在 actor backward。默认配置中
`calculate_entropy=true`，虽然 `entropy_coeff=0.0`，仍会额外计算 entropy logits，
增加训练峰值显存。

### entropy-off 重试

未修改训练数据、学习率、课程顺序或永久配置文件，仅通过命令行关闭 entropy 计算：

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-curriculum/stage-c/merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/grpo-entropy-off \
  --logger console \
  --experiment-name shopping-agent-grpo-entropy-off \
  -- actor_rollout_ref.actor.calculate_entropy=false
```

本次按要求运行到 100 个 optimizer steps。step 50 和 step 100 checkpoint 均已
生成；没有继续跑到原配置的 500 steps，也没有使用 Final-200 选择 checkpoint。

训练期间，entropy-off 版本在总序列长度 77,123、最长响应 18,063 tokens 的批次上
仍保持稳定，没有再次 OOM。训练过程的 actor 显存峰值约 51.86 GiB allocated、
69.19 GiB reserved。

最终 checkpoint 和导出的模型：

```text
outputs/models/grpo-entropy-off/global_step_100/
outputs/models/grpo-entropy-off/merged/
```

GRPO 训练日志和逐 step 诊断保留在：

```text
outputs/logs/grpo-training-entropy-off-20260814.log
outputs/models/grpo-entropy-off/training_diagnostics.jsonl
```

### GRPO Final-200 结果

step 100 merged 模型使用同一套 Final-200 Clean 协议完成 200/200 任务：

| 模型 | 正常完成 | 严格成功 / gold_purchase | 平均 reward | reward_valid |
|---|---:|---:|---:|---:|
| GRPO step 100 | 195/200（97.5%） | 125/200（62.5%） | 0.4940 | 192/200（96.0%） |

状态统计为：`done=195`、`assistant_final=1`、`error=1`、
`invalid_action_limit=3`。唯一 error 是 task `9610` 的 ContextBudgetError；
评测没有基础设施中止，剩余任务继续完成。

评测期间首次尝试曾因 ShopSimulator slot 未清空而在 task `8187` 前停止；清理
资源池后使用全新输出目录成功重跑，失败目录和日志均保留，没有混入最终结果。

vLLM 启动时还遇到 FlashInfer 与当前 CUDA/Blackwell 组合不兼容的问题。最终使用
已安装 vLLM 支持的 `VLLM_USE_FLASHINFER_SAMPLER=0`，并将编译缓存放在数据盘：

```text
/root/autodl-tmp/.cache/vllm
```

## 4. 阶段对比

| 阶段 | 严格成功率 | 平均 reward | 相对上一阶段 |
|---|---:|---:|---:|
| SFT Stage A | 55.5% | 0.3785 | — |
| SFT Stage B | 59.0% | 0.4177 | +3.5 pp |
| SFT Stage C | 61.0% | 0.4685 | +2.0 pp |
| GRPO step 100 | 62.5% | 0.4940 | +1.5 pp |

结论：主要能力增益来自 SFT 对工具协议、证据核验和多步购物流程的学习；GRPO
在 Stage C 的强初始策略上继续优化约束满足和终止决策，带来小幅但可观测的增益。
GRPO step 50 只有 checkpoint，没有跑 Final-200，因此不把它当作已测得的模型成绩。

## 5. 结果位置

SFT Final-200：

```text
outputs/evaluation/stage-a-final200/summary.json
outputs/evaluation/stage-b-final200/summary.json
outputs/evaluation/stage-c-final200-rerun/summary.json
```

GRPO step 100 Final-200：

```text
outputs/evaluation/grpo-entropy-off-step100-retry1/summary.json
outputs/evaluation/grpo-entropy-off-step100-retry1/report.html
outputs/evaluation/grpo-entropy-off-step100-retry1/trajectories.jsonl
```

以上 `outputs/` 产物均留在数据盘并被 Git 忽略；本次 GitHub 提交只包含本记录文档。
