#!/usr/bin/env bash
# V2.0 SFT 训练入口（veRL 0.8.0 fsdp_sft + ShoppingMultiTurnSFTDataset）
# 双卡 A800、BF16 全参微调。所有会影响结果的配置显式写出，不依赖 veRL 默认值。
#
# 批量结构：train_batch_size=32（全局），micro_batch_size_per_gpu=2，双卡
#           => 梯度累积 8 步/micro-batch 由 veRL 按 (32 / (2*2)) 推导。
# 检查点选择：save_freq 每 200 步存一档，max_ckpt_to_keep=3；
#           dev 集评测与最终 checkpoint 选择由独立的评测脚本执行（基于
#           outputs/split/tasks_final.jsonl 中 split=dev 的 1,050 条任务，
#           经冻结评测管线打分），训练脚本本身只监控 train loss。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m verl.trainer.sft_trainer \
  data.train_files="$ROOT/outputs/sft_dataset/train.parquet" \
  data.val_files=null \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.custom_cls.path=pkg://shopping_grpo.training.sft.multiturn_dataset \
  data.custom_cls.name=ShoppingMultiTurnSFTDataset \
  data.max_length=16384 \
  data.truncation=error \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=true \
  data.max_token_len_per_gpu=16384 \
  data.micro_batch_size_per_gpu=2 \
  data.train_batch_size=32 \
  data.num_workers=4 \
  model.path="$ROOT/models/Qwen3.5-9B" \
  model.enable_gradient_checkpointing=true \
  engine.model_dtype=bfloat16 \
  engine.dtype=bfloat16 \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  optim.optimizer=AdamW \
  optim.optimizer_impl=torch.optim \
  optim.lr=1e-5 \
  optim.weight_decay=0.01 \
  optim.lr_warmup_steps_ratio=0.03 \
  trainer.project_name=shopping-sft \
  trainer.experiment_name=v2_run1 \
  trainer.default_local_dir="$ROOT/checkpoints/sft_v2_run1" \
  trainer.total_epochs=2 \
  trainer.logger="['console']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.save_freq=200 \
  trainer.max_ckpt_to_keep=3 \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  "$@"
