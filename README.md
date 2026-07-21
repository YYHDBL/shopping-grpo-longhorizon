# Shopping SFT Data Preparation

ShopSimulator 单轮购物任务的 SFT 数据准备链路：

```text
ShopSimulator -> Teacher rollout -> 规则验收 -> OpenAI tool-calling SFT JSONL
```

当前仓库已包含 LoRA SFT 与 **Vanilla GRPO 的数据/环境最小适配准备**；尚未在仓库内固定 GRPO 超参数或启动正式 RL 训练。仍不包含 PPO、Reward Model、LLM Grader 或额外 Agent 框架。

## 目录

- `src/shopping_grpo/shop_http_env.py`：每条 trajectory 独占的结构化 ShopSimulator API 客户端。
- `src/shopping_grpo/shop_tools.py`：Teacher rollout 与未来训练共用的 OpenAI tool schema。
- `src/shopping_grpo/teacher_rollout.py`：OpenAI-compatible Teacher rollout 和断点续跑。
- `src/shopping_grpo/sft_data.py`：确定性验收和 SFT JSONL 构造。
- `scripts/`：环境 smoke、task_id 导出、采集、benchmark 与训练入口。
- `data/tasks.example.jsonl`：无隐藏信息的最小任务清单示例。
- `data/benchmarks/`：固定 held-out ShopSimulator benchmark，禁止用于训练采集。
- `docs/data_contract.md`：四类输出文件的字段约定。
- `docs/runbook.md`：端到端运行和 6000 accepted 采集方式。

## 配置

项目仅额外使用 `tqdm` 显示采集进度条；ShopSimulator 需要作为相邻仓库单独启动。

```bash
cp .env.example .env
set -a
. ./.env
set +a
```

首次运行还需安装进度条依赖：

```bash
python3 -m pip install -r requirements.txt
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

使用 DeepSeek V4 Flash 思考模式时，显式指定模型和开关。rollout 会在 raw 中保留 `reasoning_content`，以维持 Teacher 的工具调用上下文和支持后续审计；**SFT 构造默认不把它写入训练 messages**，只训练 assistant 的工具调用。若要复现 Full-CoT 消融，才显式传 `--retain-teacher-reasoning`。

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/thinking/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --model deepseek-v4-flash \
  --thinking --reasoning-effort high --limit 10 --max-steps 35
```

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/one/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --limit 1 --attempts-per-task 1 --max-steps 35

PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/ten/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --limit 10 --attempts-per-task 1 --max-steps 35

PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl --output outputs/hundred/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" --limit 100 --attempts-per-task 1 --max-steps 35
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
默认采集前 100 个任务、每条最多 35 步。下面是当前 DeepSeek V4 Pro 思考模式的 100 条命令：

```bash
PYTHONPATH=src python3 scripts/collect_sft_batch.py \
  --tasks data/shop_tasks.jsonl \
  --output-dir outputs/collection_100 \
  --base-url "$SHOPSIM_BASE_URL" \
  --model deepseek-v4-pro \
  --thinking --reasoning-effort max
```

若目标是收集固定数量的 accepted 轨迹，使用 `--target-accepted`。`--limit` 此时表示最多尝试的任务数；达到目标后立即停止。例如使用 Flash 至多尝试前 1000 个任务，收集 500 条 accepted：

运行时会显示“已扫描任务数 / 候选上限”和 `accepted=当前/目标`；中断后用完全相同的命令续跑即可。

`--workers` 可并发采集多条轨迹，但必须先把 ShopSimulator 启动为至少同样数量的环境；建议从 `--workers 4` 开始。每条轨迹仍独占一个环境，raw JSONL 由主线程顺序写入。

```bash
PYTHONPATH=src python3 scripts/collect_sft_batch.py \
  --tasks data/shop_tasks.jsonl \
  --output-dir outputs/flash_accepted_500 \
  --limit 1000 --target-accepted 500 \
  --workers 4 \
  --base-url "$SHOPSIM_BASE_URL" \
  --model deepseek-v4-flash \
  --thinking --reasoning-effort max
```

完整的 6000 accepted 续跑和验收说明见 [docs/runbook.md](docs/runbook.md)。

## 固定 benchmark 与 Base / SFT / GRPO 对比

仓库附带 `benchmark_v1`：200 个与当前 SFT 数据严格隔离的单轮 task。其指标口径和校验和见 [data/benchmarks/README.md](data/benchmarks/README.md)。Base、SFT、GRPO 必须使用完全相同的 task、工具 schema、temperature=0、max_steps=35 与 max_tokens=512。

若当前 SFT 快照更新，需要重新创建**新版本** benchmark；不要改写 v1：

```bash
PYTHONPATH=src python3 scripts/create_shop_benchmark.py \
  --tasks data/shop_tasks.jsonl \
  --sft outputs/flash_accepted_500_parallel/sft.jsonl \
  --output data/benchmarks/shop_benchmark_v2.jsonl \
  --metadata data/benchmarks/shop_benchmark_v2.metadata.json \
  --size 200 --seed 20260720
```

GPU 上启动模型的 OpenAI-compatible 服务后，运行下列命令评测。输出 raw 不进入 Git；汇总 JSON 是可比较的实验结果。

```bash
PYTHONPATH=src python3 scripts/evaluate_shop_benchmark.py \
  --benchmark data/benchmarks/shop_benchmark_v1.jsonl \
  --output outputs/eval/base_qwen35_2b/raw.jsonl \
  --summary outputs/eval/base_qwen35_2b/summary.json \
  --base-url http://127.0.0.1:5700 \
  --model Qwen/Qwen3.5-2B \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --max-tokens 512 \
  --api-key EMPTY
```

## LoRA SFT（采集完成后）

训练只使用验收后的 `sft.jsonl`，先按 `task_id` 划分，避免同题轨迹同时出现在训练与验证集。训练实现采用 `Transformers + PEFT`，并根据目标 Qwen 的 `apply_chat_template` 只计算 assistant（包括 tool call）token 的 loss；user、tool observation 不参与 loss。仓库附带当前 380 条已验收的公开训练快照；其边界和校验和见 [outputs/flash_accepted_500_parallel/README.md](outputs/flash_accepted_500_parallel/README.md)。

```bash
uv venv .venv-sft --python 3.12
source .venv-sft/bin/activate
uv pip install -r requirements-sft.txt

PYTHONPATH=src python3 scripts/split_sft_data.py \
  --input outputs/flash_accepted_500_parallel/sft.jsonl \
  --train outputs/flash_accepted_500_parallel/train.jsonl \
  --validation outputs/flash_accepted_500_parallel/validation.jsonl

PYTHONPATH=src python3 scripts/inspect_sft_data.py \
  --model Qwen/Qwen3.5-2B \
  --input outputs/flash_accepted_500_parallel/train.jsonl \
  --max-length 24576 \
  --show-example
```

需要在线查看训练曲线时，使用国内 SwanLab。先在服务器执行一次 `swanlab login`，再在训练命令中添加 `--swanlab`。它记录 Trainer 的 train/eval loss、学习率、梯度范数，以及本项目补充的单步耗时和峰值显存；日志写入本次 adapter 输出目录下的 `swanlab/`。不传该开关不会加载监控服务，也不影响训练。

预检通过后再训练。下例使用 bf16 和梯度检查点；模型或 GPU 不支持 bf16 时去掉 `--bf16`。默认 SFT 是 Action-only；对已发布的旧 Full-CoT 快照，请先由 `raw.jsonl.gz` 重建新的 Action-only `sft.jsonl`，具体命令见 [实验 03](docs/experiments/03-sft-v2-memory-and-action-only-2026-07-20.md)。

```bash
PYTHONPATH=src python3 scripts/train_lora_sft.py \
  --model Qwen/Qwen3.5-2B \
  --train outputs/flash_accepted_500_parallel/train.jsonl \
  --validation outputs/flash_accepted_500_parallel/validation.jsonl \
  --output checkpoints/qwen35-2b-shopping-lora \
  --bf16 --gradient-checkpointing \
  --swanlab --swanlab-project shopping-grpo \
  --swanlab-run-name qwen35-2b-shopping-lora-v1
```

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m py_compile src/shopping_grpo/*.py scripts/*.py
git diff --check
```
