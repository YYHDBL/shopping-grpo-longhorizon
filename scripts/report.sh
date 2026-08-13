#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:?用法: bash scripts/report.sh <评测标签>}"
RUN_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"

cd "$ROOT"
exec "$ROOT/.venv/bin/python" scripts/build_eval_report.py --run-dir "$RUN_DIR"
