#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${GRPO_MODEL_PATH:?set GRPO_MODEL_PATH to the merged SFT checkpoint or model path}"
: "${GRPO_TRAIN_FILE:?set GRPO_TRAIN_FILE to the training parquet}"
: "${GRPO_VAL_FILE:?set GRPO_VAL_FILE to the validation parquet}"
: "${GRPO_OUTPUT_DIR:?set GRPO_OUTPUT_DIR to a new checkpoint directory}"

export SHOPPING_GRPO_ROOT="$PROJECT_ROOT"
# 不继承旧 shell 中可能指向 reference fork 的 PYTHONPATH。
export PYTHONPATH="$PROJECT_ROOT/src"
export SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"

cd "$PROJECT_ROOT"

python3 "$PROJECT_ROOT/scripts/generate_verl_shop_configs.py" \
  --tool-output "$PROJECT_ROOT/configs/verl/shop_tools.json"

python3 "$PROJECT_ROOT/scripts/check_grpo_runtime.py" "$@"

exec python3 -m verl.trainer.main_ppo \
  --config-path="$PROJECT_ROOT/configs/verl" \
  --config-name=vanilla_grpo \
  "$@"
