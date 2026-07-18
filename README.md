# Shopping SFT Data Preparation

ShopSimulator 单轮购物任务的 SFT 数据准备链路：

```text
ShopSimulator -> Teacher rollout -> 规则验收 -> OpenAI tool-calling SFT JSONL
```

当前仓库只负责收集和构造 SFT-ready 数据。不包含 LoRA SFT、GRPO、PPO、Reward Model、LLM Grader 或额外 Agent 框架。

## 目录

- `src/shopping_grpo/shop_http_env.py`：每条 trajectory 独占的结构化 ShopSimulator API 客户端。
- `src/shopping_grpo/shop_tools.py`：Teacher rollout 与未来训练共用的 OpenAI tool schema。
- `src/shopping_grpo/teacher_rollout.py`：OpenAI-compatible Teacher rollout 和断点续跑。
- `src/shopping_grpo/sft_data.py`：确定性验收和 SFT JSONL 构造。
- `scripts/`：环境 smoke、task_id 导出、采集和数据构造入口。
- `data/tasks.example.jsonl`：无隐藏信息的最小任务清单示例。
- `docs/data_contract.md`：四类输出文件的字段约定。
- `docs/runbook.md`：端到端运行和 6000 accepted 采集方式。

## 配置

项目只依赖 Python 标准库。ShopSimulator 需要作为相邻仓库单独启动。

```bash
cp .env.example .env
set -a
. ./.env
set +a
```

`.env` 不会被 Python 自动读取；上述命令将它导出到当前 shell。不要提交 `.env`。

在另一终端，从本仓库根目录启动一个单环境 ShopSimulator 服务：

```bash
cd ../ShopSimulator/shop_env/shop_env
PYTHONPATH=.. SHOPSIM_ALLOW_LINEAR_SEARCH=1 \
  ../.venv-clean/bin/python -u -c \
  'import pack_api; pack_api.env_max_num=1; pack_api.initialize_environments(); pack_api.app.run(host="127.0.0.1", port=5000)'
```

若 ShopSimulator 尚未有可用环境，先在其 `shop_env/` 下新建干净环境再安装依赖；不要修补旧的损坏环境。

回到本仓库，先验证环境接口：

```bash
PYTHONPATH=src python3 scripts/smoke_shop_env.py \
  --base-url "$SHOPSIM_BASE_URL" --task-id 0 \
  --actions 'search[乳胶枕]'
```

## 采集与构造

先导出完整 task_id 清单。脚本直接调用 ShopSimulator 当前的数据清洗和 goal 生成代码，保证 id 的顺序与环境一致；必须使用 ShopSimulator 的干净虚拟环境运行。每行只含公开任务 id；环境在 `reset` 后提供用户需求，因此不需要把 goal、标准答案或 reward 写入任务文件。

```bash
../ShopSimulator/shop_env/.venv-clean/bin/python scripts/export_shop_task_ids.py \
  --shopsim-root ../ShopSimulator/shop_env \
  --output data/shop_tasks.jsonl
```

若已存在旧的本地清单，确认替换时添加 `--force`。`data/tasks.example.jsonl` 只保留作最小测试示例。

1、10、100 个任务分别只需更换 `--limit`。以下命令的 `--limit` 限制任务数，不保证同样数量的 accepted 轨迹。

使用 DeepSeek V4 Flash 思考模式时，显式指定模型和开关。rollout 会在 raw 中保留并回传 `reasoning_content` 以支持 tool calling；构造 SFT JSONL 时会自动移除该字段。

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/thinking/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --model deepseek-v4-flash \
  --thinking --reasoning-effort high --limit 10 --max-steps 50
```

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/one/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --limit 1 --attempts-per-task 1 --max-steps 50

PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/ten/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --limit 10 --attempts-per-task 1 --max-steps 50

PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/hundred/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --limit 100 --attempts-per-task 1 --max-steps 50
```

将任一 raw JSONL 构造成 accepted、rejected、统计和标准 OpenAI messages JSONL：

```bash
PYTHONPATH=src python3 scripts/build_sft_data.py \
  --raw outputs/hundred/raw.jsonl \
  --accepted outputs/hundred/accepted.jsonl \
  --rejected outputs/hundred/rejected.jsonl \
  --stats outputs/hundred/stats.json \
  --sft outputs/hundred/sft_openai_messages.jsonl
```

推荐使用批次脚本采集并自动重建上述四类文件。相同 `--output-dir` 可直接断点续跑；
默认采集前 100 个任务、每条最多 50 步。下面是当前 DeepSeek V4 Pro 思考模式的 100 条命令：

```bash
PYTHONPATH=src python3 scripts/collect_sft_batch.py \
  --tasks data/shop_tasks.jsonl \
  --output-dir outputs/collection_100 \
  --base-url "$SHOPSIM_BASE_URL" \
  --model deepseek-v4-pro \
  --thinking --reasoning-effort max
```

完整的 6000 accepted 续跑和验收说明见 [docs/runbook.md](docs/runbook.md)。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m py_compile src/shopping_grpo/*.py scripts/*.py
git diff --check
```
