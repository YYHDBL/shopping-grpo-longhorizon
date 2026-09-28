#!/usr/bin/env bash
# V2.0 SFT 训练入口（veRL 0.9.1 fsdp_sft + ShoppingMultiTurnSFTDataset）
# 双卡 A800、BF16 全参微调。所有会影响结果的配置显式写出，不依赖 veRL 默认值。
#
# 批量结构：train_batch_size=64（全局）/双卡=32 条/卡，动态 bsz 按 16384 token/卡切块，
#           每卡约 8~11 块做梯度累积后更新一次；micro_batch_size_per_gpu=4 为单块条数硬上限；
#           offload 全关（显存吃满），PYTORCH_CUDA_ALLOC_CONF=expandable_segments 防碎片（launch 脚本注入）。
# 检查点：save_freq=50 与验证同步（50/100/150/最终共 4 档，每档约 90GB），max_ckpt_to_keep=3 滚动保留+final；
#           dev 集评测与最终 checkpoint 选择由独立的评测脚本执行（基于
#           outputs/split/tasks_final.jsonl 中 split=dev 的 1,050 条任务，
#           经冻结评测管线打分）。训练同步记录优化、Agentic token、序列长度、
#           吞吐、耗时和资源指标到 console 与 SwanLab。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$ROOT/outputs/swanlog}"

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m shopping_grpo.training.sft.trainer \
  data.train_files="$ROOT/outputs/sft_dataset/train.parquet" \
  data.val_files="$ROOT/outputs/sft_dataset/val.parquet" \
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
  data.micro_batch_size_per_gpu=4 \
  data.train_batch_size=64 \
  data.num_workers=0 \
  model.path="$ROOT/models/Qwen3.5-9B" \
  model.trust_remote_code=true \
  +model.override_config.attn_implementation=sdpa \
  engine.model_dtype=bfloat16 \
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
  trainer.logger="['console','swanlab']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.save_freq=50 \
  trainer.max_ckpt_to_keep=3 \
  trainer.test_freq=50 \
  trainer.resume_mode=auto \
  "$@"
