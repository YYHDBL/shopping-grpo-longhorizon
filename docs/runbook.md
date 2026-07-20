# 运行手册

在本仓库根目录执行以下命令；先按 README 导出 `.env` 并启动 ShopSimulator。

## 任务清单

完整任务清单不要手写。用下列命令从 ShopSimulator 实际使用的商品数据和 goal 构造逻辑导出，生成的 id 从 0 连续递增，顺序与环境一致：

```bash
../ShopSimulator/shop_env/.venv-clean/bin/python scripts/export_shop_task_ids.py \
  --shopsim-root ../ShopSimulator/shop_env \
  --output data/shop_tasks.jsonl
```

导出器会保护已有文件；确认覆盖时使用 `--force`。采集器接受顶层 `task_id`，也兼容旧格式 `extra_info.interaction_kwargs.task_id`。一行可以包含公开 `prompt`，但环境 reset 返回的用户需求仍会被追加。任务文件中绝不能写入 goal、答案或 reward 元数据。

## 可断点续跑的采集

`--attempts-per-task N` 为每个任务定义 `0` 到 `N - 1` 的尝试编号。raw 输出中已有 `(task_id, attempt_index)` 的记录会在下一次运行时跳过；没有 `attempt_index` 的旧记录视为 attempt 0。

日常采集使用 `scripts/collect_sft_batch.py`。它在一个目录中维护 raw、accepted、rejected、统计和 SFT 文件；中断后重跑同一命令即可续跑，并从完整 raw 重建派生产物。例如先采集 100 个任务：

首次执行先安装终端进度条依赖：

```bash
python3 -m pip install -r requirements.txt
```

```bash
PYTHONPATH=src python3 scripts/collect_sft_batch.py \
  --tasks data/shop_tasks.jsonl \
  --output-dir outputs/collection_100 \
  --base-url "$SHOPSIM_BASE_URL" \
  --model deepseek-v4-pro \
  --thinking --reasoning-effort max
```

需要“收满 N 条 accepted”时，添加 `--target-accepted N`；`--limit` 只是候选任务上限，达到 accepted 目标即停止。例如 500 条目标可先给 1000 个候选任务，并根据实际通过率增加上限：

进度条的主进度表示已扫描任务数，右侧 `accepted=当前/目标` 表示已经通过确定性验收的轨迹数。按 `Ctrl+C` 可安全中断当前轨迹并释放环境；之后原样重跑即可续跑。

采集器支持 `--workers N` 并发 rollout。先让 ShopSimulator 初始化至少 `N` 个环境，再使用同样的 `N`；建议从 4 开始。每个 worker 通过 reset 获得独占环境，主线程统一追加 raw JSONL，因此不会混淆轨迹或破坏断点续跑。

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

目标为 6000 条 accepted 时，任务数应显著多于 6000，并设定固定尝试次数。当前环境可导出 23,421 个 task_id；例如前 8,000 个任务、每个任务 4 次尝试，最多产生 32,000 条 raw trajectory。

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl \
  --output outputs/accepted_6000/raw.jsonl \
  --base-url "$SHOPSIM_BASE_URL" \
  --attempts-per-task 4 --temperature 0.8 --top-p 1.0 --max-steps 35

PYTHONPATH=src python3 scripts/build_sft_data.py \
  --raw outputs/accepted_6000/raw.jsonl \
  --accepted outputs/accepted_6000/accepted.jsonl \
  --rejected outputs/accepted_6000/rejected.jsonl \
  --stats outputs/accepted_6000/stats.json \
  --sft outputs/accepted_6000/sft_openai_messages.jsonl
```

采集中断后，原样重跑采集命令即可：它只追加缺失的 task-attempt 对；之后重跑构造器，它会从完整 raw 文件重新生成四类派生产物。

每次构造后读取 `outputs/accepted_6000/stats.json`，在 `accepted >= 6000` 时停止。如果已计划的尝试全部完成仍未达到目标，增加 task_id 范围或把尝试次数从 4 提升到 8；已有结果不会变化，只会新增 attempt 4 到 7。

每个采样配置使用独立输出目录，以便追溯模型、温度、任务来源和重试策略，无需额外采集框架。

## 固定 benchmark：Base、SFT 与 GRPO 的统一评测

先启动 ShopSimulator 结构化服务；评测器只通过 `/api/shop_agent` 操作环境，不读取 HTML reward。仓库中的 `data/benchmarks/shop_benchmark_v1.jsonl` 是 200 条固定 held-out task，和当前 SFT 的 380 条 task 无重叠。不得用这 200 条 task 继续采集训练轨迹。

所有对比实验必须保持下列设置不变：task 清单、工具 schema、system prompt、temperature=0、max_steps=35、max_tokens=512、每 task 一次 rollout。`max_tokens` 是单次模型生成上限，避免未调用工具时持续生成纯文本；它不是模型上下文长度。严格成功率以全部 200 条 task 为分母，要求购买后环境确认结束，且 `r_type`、`r_att`、`r_option`、`r_price` 均为 1。

启动 GPU 模型服务后，Base、SFT adapter 和 GRPO checkpoint 分别仅替换 `--model`、`--llm-base-url` 与输出目录，运行相同命令：

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

`raw.jsonl` 支持断点续跑；`summary.json` 记录严格成功率、四项分量、平均步数、状态和动作守卫原因。先保存 Base 结果，再进行 SFT 与 GRPO，避免事后改变评测协议。

## LoRA SFT 冷启动

当 accepted 轨迹收集完成后，只训练同目录的 `sft.jsonl`。不要把 raw、rejected、goal、reward_detail 或环境隐藏信息喂给模型。先按 task_id 划分，以防同一个任务的多次尝试泄漏到验证集：

```bash
uv venv .venv-sft --python 3.12
source .venv-sft/bin/activate
uv pip install -r requirements-sft.txt

PYTHONPATH=src python3 scripts/split_sft_data.py \
  --input outputs/flash_accepted_500_parallel/sft.jsonl \
  --train outputs/flash_accepted_500_parallel/train.jsonl \
  --validation outputs/flash_accepted_500_parallel/validation.jsonl \
  --validation-ratio 0.05
```

### 训练监控（可选）

本项目使用 SwanLab，不依赖 W&B。首次在线使用前在服务器执行 `swanlab login`；训练时加 `--swanlab` 即可。SwanLab 由 Transformers 原生集成，自动记录 train/eval loss、learning rate、grad norm 和硬件信息；训练脚本额外写入 step time 与峰值显存。日志固定保存到本次 `--output` 目录的 `swanlab/`，未启用时训练不受影响。若只想保存本地日志，将 `--swanlab-mode local` 传入训练命令。

训练前必须使用目标模型的 chat template 做预检；输出的 `assistant_label_preview` 应只包含 assistant 的文本和 tool call，不应包含用户需求或 tool observation：

```bash
PYTHONPATH=src python3 scripts/inspect_sft_data.py \
  --model Qwen/Qwen3.5-2B \
  --input outputs/flash_accepted_500_parallel/train.jsonl \
  --max-length 24576 \
  --show-example
```

预检通过后执行 LoRA SFT。默认 LoRA rank 为 16、训练 3 epoch；产物是 adapter，不覆盖 base model。中断后可通过 `--resume-from-checkpoint checkpoints/.../checkpoint-*` 续训：

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
