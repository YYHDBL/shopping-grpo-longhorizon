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

```bash
PYTHONPATH=src python3 scripts/collect_sft_batch.py \
  --tasks data/shop_tasks.jsonl \
  --output-dir outputs/flash_accepted_500 \
  --limit 1000 --target-accepted 500 \
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
  --attempts-per-task 4 --temperature 0.8 --top-p 1.0 --max-steps 50

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
