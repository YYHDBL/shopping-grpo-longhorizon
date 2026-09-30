#!/bin/bash
# GRPO run2 正式跑启动脚本（2026-09-30 定稿配置，用户指令后执行）
# 刹车组合：KL 锚 0.01 + lr 2e-7 cosine + batch 16×n8 + DAPO 动态采样 + IS 修正
# 前置：环境服务 160 槽已起（容器内 5700）；预计 60~75h（含动态采样补采）
set -e
SHOPPING_GRPO_ROOT=/data/jyh-yyh/shopping-grpo-longhorizon
cd $SHOPPING_GRPO_ROOT

export CUDA_VISIBLE_DEVICES=GPU-9f79d35e-399f-179f-159e-cdd154121753,GPU-718811c8-ce9b-ec96-55db-0c626a26d812,GPU-665edded-b039-f90a-7546-71d31e327cb5,GPU-0e5ffba6-80f2-2cee-9717-29c25a8add18
export SHOPPING_GRPO_ROOT
export GRPO_MODEL_PATH=$SHOPPING_GRPO_ROOT/outputs/models/sft-merged
export GRPO_TRAIN_FILE=$SHOPPING_GRPO_ROOT/outputs/grpo/train_1000.parquet
export GRPO_VAL_FILE=$SHOPPING_GRPO_ROOT/outputs/grpo/smoke.parquet
export GRPO_ROLLOUT_DIR=$SHOPPING_GRPO_ROOT/outputs/rollout_data/run2
export GRPO_OUTPUT_DIR=$SHOPPING_GRPO_ROOT/checkpoints/grpo_run2
export SHOPSIM_BASE_URL=http://127.0.0.1:5700
export PYTHONPATH=$SHOPPING_GRPO_ROOT/src:$SHOPPING_GRPO_ROOT
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=$(grep -E '^SWANLAB_API_KEY=' .env | cut -d= -f2)

mkdir -p $GRPO_OUTPUT_DIR $GRPO_ROLLOUT_DIR
python3 -m verl.trainer.main_ppo --config-name grpo_run2 --config-dir $SHOPPING_GRPO_ROOT/configs 2>&1 | tee $SHOPPING_GRPO_ROOT/outputs/grpo_run2.log
