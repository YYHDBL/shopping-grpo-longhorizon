#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sft_python="${SFT_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${sft_python}" ]]; then
  echo "找不到 Python 环境：${sft_python}；请先运行 scripts/setup.sh 或设置 SFT_PYTHON。" >&2
  exit 2
fi
exec "${sft_python}" "${repo_root}/scripts/run_sft_curriculum.py" "$@"
