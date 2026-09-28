#!/usr/bin/env bash
# V2.0 SFT 训练启动脚本（veRL fsdp_sft + 自定义购物轨迹数据集）
# 双卡用法：bash scripts/train_sft.sh
# 关键参数在下方 HYDRA 覆盖里，max_length 依据 token 统计定为 16384（丢弃超长约 4%）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:$ROOT:$PYTHONPATH"

torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m verl.trainer.sft_trainer \
  data.train_files="$ROOT/outputs/sft_dataset/train.parquet" \
  data.val_files=null \
  data.max_length=16384 \
  data.truncation=error \
  data.custom_cls.path=shopping_grpo.training.sft.multiturn_dataset \
  data.custom_cls.name=ShoppingMultiTurnSFTDataset \
  data.micro_batch_size_per_gpu=2 \
  data.max_token_len_per_gpu=16384 \
  model.path="$ROOT/models/Qwen3.5-9B" \
  model.enable_gradient_checkpointing=true \
  trainer.project_name=shopping-sft \
  trainer.experiment_name=v2_run1 \
  trainer.total_epochs=2 \
  trainer.logger="['console']" \
  trainer.n_gpus_per_node=2 \
  trainer.save_freq=100 \
  "$@"
