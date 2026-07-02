# Stage 1 Smoke Test Runbook

This runbook verifies that a local ShopSimulator HTTP app can be reached by the
project smoke client.

## Current Local Setup

ShopSimulator upstream expects a Lucene index at:

```bash
/Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env/search_engine/indexes
```

The local clone does not include `search_engine`, so Stage 1 uses the temporary
linear-search fallback guarded by:

```bash
SHOPSIM_ALLOW_LINEAR_SEARCH=1
```

This is for local smoke only. Server training should use the real Lucene index
if it is available.

## Environment

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env
uv venv --python 3.10 .venv
uv pip install 'gym==0.24.0' 'flask==2.1.2' 'werkzeug<3' 'beautifulsoup4==4.11.1' 'numpy>=1.23,<2' 'pyserini==0.17.0' 'rich==12.4.4' 'spacy==3.7.5' 'thefuzz==0.19.0' 'torch>=2.2' 'tqdm==4.64.0' 'requests==2.27.1' 'selenium==4.2.0' faiss-cpu
uv pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl
gzip -dc data/fine_items_eval_train_all.json.gz > data/items_eval_train.json
```

The full upstream requirements currently pull an old `tokenizers==0.12.1` path
that fails locally, so the command above installs only the dependencies needed
for Stage 1 smoke.

## Start App

Ports `5000` and `7000` are occupied by macOS ControlCenter on this machine.
Use `7001` for the light Flask app:

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/ShopSimulator/shop_env
PYTHONPATH=. SHOPSIM_ALLOW_LINEAR_SEARCH=1 SHOPSIM_APP_PORT=7001 .venv/bin/python web_agent_site/app.py
```

Expected listener:

```text
Running on http://127.0.0.1:7001
```

## Run Smoke

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon
python3 scripts/smoke_shop_env.py --base-url http://127.0.0.1:7001 --task-id 0 --actions 'search[乳胶枕]'
```

Expected output is a JSON path under `outputs/smoke/`.

## Verified Result

Last verified output:

```bash
outputs/smoke/task_0000_20260702_230627.json
```

It contains two HTTP steps:

```text
start -> 200
search[乳胶枕] -> 200
```
