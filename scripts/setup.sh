#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${ROOT}/environments/ShopSimulator"
SHOP_ENV_ROOT="${SHOPSIM_ROOT}/shop_env"
ENV_DIR="${SHOPSIM_ROOT}/.venv-shopsim"
COMPRESSED_PRODUCTS="${SHOP_ENV_ROOT}/data/fine_items_eval_train_all.json.gz"
PRODUCTS="${SHOP_ENV_ROOT}/data/items_eval_train.json"
EXPECTED_PRODUCT_SHA256="57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
MAIN_PYTHON="${MAIN_PYTHON:-3.12}"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-3.12}"

if [[ ! -f "${COMPRESSED_PRODUCTS}" ]]; then
  echo "Missing embedded product archive: ${COMPRESSED_PRODUCTS}" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the isolated ShopSimulator environment" >&2
  exit 1
fi

cd "${ROOT}"
uv sync --python "${MAIN_PYTHON}" --extra sft --extra grpo

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  uv venv --python "${SHOPSIM_PYTHON}" "${ENV_DIR}"
fi
uv pip install \
  --python "${ENV_DIR}/bin/python" \
  -r "${SHOP_ENV_ROOT}/requirements.txt"

if [[ ! -f "${PRODUCTS}" ]]; then
  temporary_products="${PRODUCTS}.preparing"
  trap 'rm -f "${temporary_products}"' EXIT
  gzip -cd "${COMPRESSED_PRODUCTS}" > "${temporary_products}"
  actual_sha256="$(sha256sum "${temporary_products}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
    echo "Product data SHA-256 mismatch: ${actual_sha256}" >&2
    exit 1
  fi
  mv "${temporary_products}" "${PRODUCTS}"
  trap - EXIT
fi

actual_sha256="$(sha256sum "${PRODUCTS}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
  echo "Existing product data SHA-256 mismatch: ${actual_sha256}" >&2
  exit 1
fi

cd "${SHOP_ENV_ROOT}"
PYTHONPATH=. "${ENV_DIR}/bin/python" scripts/build_index.py

cd "${ROOT}"

echo "Shopping Agent training environment is ready."
echo "Product SHA-256: ${actual_sha256}"
echo "Index: ${SHOP_ENV_ROOT}/search_engine/products.sqlite3"
