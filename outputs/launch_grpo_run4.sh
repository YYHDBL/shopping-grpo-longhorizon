#!/bin/bash
# GRPO run3 正式跑启动脚本（2026-10-01，curriculum 版）
# 数据：同一批 1000 题（隔离性不变），顺序按用户方案重排：
#   [常规难度 easy/medium/hard 混合打乱 700 条] → [teacher_hard 打乱 300 条]（课程学习）
# 生成脚本 scripts/reorder_grpo_curriculum.py（含自校验）。
# 配置变化集中在刹车参数：batch 32×n8=256 轨迹、lr 5e-7 cosine(min 0.4)、KL 0.003。
# 环境槽位 256（已就位）。
set -e
SHOPPING_GRPO_ROOT=/data/jyh-yyh/shopping-grpo-longhorizon
cd $SHOPPING_GRPO_ROOT

export CUDA_VISIBLE_DEVICES=GPU-9f79d35e-399f-179f-159e-cdd154121753,GPU-718811c8-ce9b-ec96-55db-0c626a26d812,GPU-665edded-b039-f90a-7546-71d31e327cb5,GPU-0e5ffba6-80f2-2cee-9717-29c25a8add18
export SHOPPING_GRPO_ROOT
export GRPO_MODEL_PATH=$SHOPPING_GRPO_ROOT/outputs/models/sft-merged
export GRPO_TRAIN_FILE=$SHOPPING_GRPO_ROOT/outputs/grpo/train_1000_curriculum.parquet
export GRPO_VAL_FILE=$SHOPPING_GRPO_ROOT/outputs/grpo/smoke.parquet
export GRPO_ROLLOUT_DIR=$SHOPPING_GRPO_ROOT/outputs/rollout_data/run3
export GRPO_OUTPUT_DIR=$SHOPPING_GRPO_ROOT/checkpoints/grpo_run4
export SHOPSIM_BASE_URL=http://127.0.0.1:5700
export PYTHONPATH=$SHOPPING_GRPO_ROOT/src:$SHOPPING_GRPO_ROOT
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=$(grep -E '^SWANLAB_API_KEY=' .env | cut -d= -f2)

mkdir -p $GRPO_OUTPUT_DIR $GRPO_ROLLOUT_DIR
python3 -m verl.trainer.main_ppo --config-name grpo_run4 --config-dir $SHOPPING_GRPO_ROOT/configs trainer.resume_mode=auto 2>&1 | tee $SHOPPING_GRPO_ROOT/outputs/grpo_run4.log
