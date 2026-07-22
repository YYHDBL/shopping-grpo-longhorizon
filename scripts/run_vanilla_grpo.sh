#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${GRPO_MODEL_PATH:?set GRPO_MODEL_PATH to the merged SFT checkpoint or model path}"
: "${GRPO_TRAIN_FILE:?set GRPO_TRAIN_FILE to the training parquet}"
: "${GRPO_VAL_FILE:?set GRPO_VAL_FILE to the validation parquet}"
: "${GRPO_OUTPUT_DIR:?set GRPO_OUTPUT_DIR to a new checkpoint directory}"

export SHOPPING_GRPO_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"

# Prefer the checked-in sibling reference checkout when it exists. Its veRL fork
# imports one helper from the adjacent project root, so both paths are required.
REFERENCE_ROOT="$PROJECT_ROOT/../agentic-grpo-longhorizon"
if [[ -d "$REFERENCE_ROOT/verl/verl" ]]; then
  export PYTHONPATH="$REFERENCE_ROOT/verl:$REFERENCE_ROOT/agentic-grpo-longhorizon:$PYTHONPATH"
fi

cd "$PROJECT_ROOT"

python3 "$PROJECT_ROOT/scripts/generate_verl_shop_configs.py" \
  --tool-output "$PROJECT_ROOT/configs/verl/shop_tools.json" \
  --interaction-output "$PROJECT_ROOT/configs/verl/shop_interaction.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --max-steps 35

python3 "$PROJECT_ROOT/scripts/check_grpo_runtime.py"

exec python3 -m verl.trainer.main_ppo \
  --config-path="$PROJECT_ROOT/configs/verl" \
  --config-name=vanilla_grpo \
  "$@"
