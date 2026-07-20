# 实验 03：SFT v2 — 长上下文显存与 Action-only 决策

> 状态：待运行｜日期：2026-07-20｜目标：在 RTX 4090 48GB 上以最小工程改动验证 16K、20K、24K SFT 是否可训练。

## 问题与假设

- **问题：** v1 在 12,288 token 才能训练，丢弃了大量长轨迹；24K 在反向传播阶段 OOM。
- **假设 1：** 只监督 assistant token 但仍物化全序列 LM Head logits，是 v1 的主要额外峰值显存来源；Liger 的 fused linear cross-entropy 能显著缓解。
- **假设 2：** 不训练 DeepSeek Teacher 的逐步思考，而只学习工具调用，可减少无关上下文和“过度探索”模仿；Shopping Agent 不需要向用户展示思考过程。
- **假设 3：** 本机 PyTorch 的 SDPA 已是可用的注意力后端，不应为编译失败的 FlashAttention 2 改动 CUDA 环境。

## 已知事实与取舍

Qwen3.5-2B 支持 thinking，但默认 non-thinking；其 2B 版本在 thinking 模式可能出现循环。当前固定 benchmark 也是 non-thinking。因此新训练数据默认 **Action-only**：Teacher 的 `reasoning_content` 留在 raw trajectory 供审计，但不会写入 SFT assistant labels。

FlashAttention 2 因服务器 PyTorch CUDA 13.0 与系统 CUDA 12.8 不匹配而无法可靠安装；继续 patch 编译检查没有工程价值。本实验改用 Transformers 的 `attn_implementation="sdpa"`，由 PyTorch 选择可用的 memory-efficient/Flash SDP 内核。它不是 FA2 的逐字等价物，必须记录实测峰值显存和耗时，不能预设收益。

不在本轮做的事：迁移 TRL、长轨迹窗口化、导出 text-only checkpoint、改 benchmark 协议。它们会改变数据或训练语义，不适合作为 OOM 的第一修复。

## 数据版本

仓库发布的 `outputs/flash_accepted_500_parallel/sft.jsonl` 是历史 Full-CoT 快照，不能直接用于本实验。应从同目录已发布的 `raw.jsonl.gz` 重建一个新目录，原始快照不改写：

```bash
mkdir -p outputs/flash_accepted_500_parallel_action_only
gzip -dc outputs/flash_accepted_500_parallel/raw.jsonl.gz \
  > outputs/flash_accepted_500_parallel_action_only/raw.jsonl

PYTHONPATH=src python3 scripts/build_sft_data.py \
  --raw outputs/flash_accepted_500_parallel_action_only/raw.jsonl \
  --accepted outputs/flash_accepted_500_parallel_action_only/accepted.jsonl \
  --rejected outputs/flash_accepted_500_parallel_action_only/rejected.jsonl \
  --stats outputs/flash_accepted_500_parallel_action_only/reject_stats.json \
  --sft outputs/flash_accepted_500_parallel_action_only/sft.jsonl

PYTHONPATH=src python3 scripts/split_sft_data.py \
  --input outputs/flash_accepted_500_parallel_action_only/sft.jsonl \
  --train outputs/flash_accepted_500_parallel_action_only/train.jsonl \
  --validation outputs/flash_accepted_500_parallel_action_only/validation.jsonl
```

不传 `--retain-teacher-reasoning` 即为 Action-only；`--retain-teacher-reasoning` 仅用于后续 Full-CoT 对照，不能覆盖上述目录。

## 环境与统一 smoke 命令

```bash
source .venv-sft/bin/activate
uv pip install -r requirements-sft-accelerated.txt

export MODEL=Qwen/Qwen3.5-2B
export DATA=outputs/flash_accepted_500_parallel_action_only
export COMMON="--model $MODEL --train $DATA/train.jsonl --validation $DATA/validation.jsonl \
  --bf16 --gradient-checkpointing --epochs 1 --max-steps 2 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --swanlab --swanlab-project shopping-grpo --swanlab-mode local"
```

`max_steps=2` 只用于测峰值显存、后端是否可运行和单步时间，不用于评价模型效果。每次运行前确保没有 vLLM 占用 GPU，并保留各自输出目录的 `train_summary.json` 和 SwanLab 本地日志。

## A/B/C/D 实验矩阵

| 组别 | 目的 | 额外开关 | 长度顺序 | 成功条件 |
|---|---|---|---|---|
| A | Action-only BF16 LoRA 基线 | 无 | 12K | 作为新数据版本的显存参考 |
| B | 验证 fused loss | `--liger-kernel` | 12K → 16K | 不改变 loss 数值语义，峰值低于 A 或能稳定训练更长序列 |
| C | 验证 PyTorch SDPA | B + `--attention-implementation sdpa` | 16K → 20K → 24K | 记录真实后端、显存与 step 时间；不要求一定优于 B |
| D | 最后释放基座权重空间 | C + `--qlora` | C 首个 OOM 长度起 | 正常完成 2 step；仅在 B/C 仍 OOM 时运行 |

```bash
# A
PYTHONPATH=src python3 scripts/train_lora_sft.py $COMMON \
  --max-length 12288 --output checkpoints/smoke-a-action-only-12k

# B：先 12K，再把 --max-length 改为 16384
PYTHONPATH=src python3 scripts/train_lora_sft.py $COMMON \
  --liger-kernel --max-length 12288 --output checkpoints/smoke-b-liger-12k

# C：按 16K、20K、24K 顺序运行；任一长度 OOM 后停止向上尝试
PYTHONPATH=src python3 scripts/train_lora_sft.py $COMMON \
  --liger-kernel --attention-implementation sdpa \
  --max-length 16384 --output checkpoints/smoke-c-liger-sdpa-16k

# D：只在 C 的相同长度 OOM 时运行
PYTHONPATH=src python3 scripts/train_lora_sft.py $COMMON \
  --liger-kernel --attention-implementation sdpa --qlora \
  --max-length 24576 --output checkpoints/smoke-d-qlora-24k
```

## 记录表（运行后填写）

| 组别 | max_length | kept train/val | 是否完成 2 step | 峰值 GiB | step 秒数 | 结论 |
|---|---:|---:|---|---:|---:|---|
| A | 12,288 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| B | 12,288 / 16,384 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| C | 16,384 / 20,480 / 24,576 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| D | 由 C 的 OOM 点决定 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

完成 smoke 后，选择“能稳定训练且保留样本最多”的配置做 3 epoch 正式 SFT；再用固定 `benchmark_v1` 做一次完整 200 条评测。不得因为结果好坏修改 benchmark、system prompt、工具 schema、temperature、max_steps 或 max_tokens。
