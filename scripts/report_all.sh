#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATION_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation}"

for run_dir in "$EVALUATION_DIR"/*; do
  [[ -f "$run_dir/summary.json" && -f "$run_dir/trajectories.jsonl" ]] || continue
  "$ROOT/.venv/bin/python" "$ROOT/scripts/build_eval_report.py" --run-dir "$run_dir"
done

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/build_comparison_report.py" \
  --evaluation-dir "$EVALUATION_DIR"
