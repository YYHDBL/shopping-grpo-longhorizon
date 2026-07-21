# Vanilla GRPO v1：任务冻结与 veRL 接入准备

> 状态：准备完成，尚未在 GPU 上启动训练。本文只固定不会因超参数而变化的实验资产；group size、并发、学习率与 KL 系数留给服务器 smoke test 决定。

## 目标

SFT v2 已将 `benchmark_v2_50` 的严格成功率从 Base 的 0% 提升到 12%，但 68% 轨迹仍走满 35 步。下一步先跑最小 Vanilla GRPO：不引入 LLM Judge、Reward Model、PRM-Lite、长度塑形或 Persona，只优化 ShopSimulator 的原生终局 reward。

## 已冻结的边界

| 资产 | 决策 |
|---|---|
| 评测 | `data/benchmarks/shop_benchmark_v2_50.jsonl`，严格取 v1 的前 50 条 |
| RL 候选池 | `data/splits/grpo_probe_pool_v1.jsonl`，2,000 个未见 task |
| 排除集 | 已发布 Teacher raw rollout 全部 757 个 task + benchmark v2_50 |
| 最终训练集 | 1,000 个新 task，不是 1,000 条离线轨迹 |
| 长度分层 | short/medium/long = 300/450/250，按 **SFT policy probe 实测工具步数** 划分 |
| reward | 仅正常 `done && over` 时的 ShopSimulator `final_reward`；其他终止均为 0 |

为什么不按 Teacher 轨迹长度分层：GRPO 面对的是 SFT policy 的真实困难度；Teacher 的探索方式与它不同。为什么先跑 2,000 个 probe：现有 SFT 约 68% 会走满步数，必须有足够余量才能保证三桶都能精确抽满；桶不足时宁可停止，不制造“表面均衡”的数据。

## SFT → GRPO 的权重边界

保留原始 Base 与 SFT adapter 不动。先将 SFT LoRA 合并到一个新目录，再以这个 merged checkpoint 作为 GRPO 初始策略，并挂载一枚新的 GRPO LoRA。这样 SFT 与 RL 的贡献可分别保存、回滚和比较。

```bash
PYTHONPATH=src python3 scripts/merge_lora_adapter.py \
  --base-model /path/to/Qwen3.5-2B \
  --adapter checkpoints/qwen35-2b-shopping-lora-v2 \
  --output checkpoints/qwen35-2b-shopping-sft-v2-merged \
  --bf16
```

## veRL 最小适配层

`src/shopping_grpo/verl_adapter/` 只做四件事：

1. `Interaction.start_interaction` 对每条 rollout 调 `reset(task_id)`，取得独占 HTTP 环境；
2. `Tool` 复用仓库唯一的 `SHOP_TOOL_SCHEMAS`、`tool_call_to_action` 和动作守卫；
3. `calculate_score` 只返回环境原生终局 reward；
4. `finalize_interaction` 无条件 `release()`，避免单 env slot 被占死。

veRL 的 prompt 必须在训练开始前准备成 parquet。`task_id` 本身不含用户需求；`scripts/prepare_verl_grpo_dataset.py` 会先 reset 一次，只提取用户可见 instruction 写入 prompt，绝不写入 goal、标准答案或 reward_detail。训练时 Interaction 会再次 reset 同一 task，保证每个 group sample 独占环境。

## 服务器执行顺序

1. 用 merged SFT 模型跑 `grpo_probe_pool_v1.jsonl`，沿用 `evaluate_shop_benchmark.py`、`temperature=0`、`max_steps=35`，保存 raw probe；
2. 分层冻结 1,000 个 task：

```bash
PYTHONPATH=src python3 scripts/prepare_grpo_tasks.py select \
  --probes outputs/grpo_probe_sft_v2/raw.jsonl
```

3. 启动 ShopSimulator 后转 parquet：

```bash
PYTHONPATH=src python3 scripts/prepare_verl_grpo_dataset.py \
  --tasks data/splits/grpo_train_v1.jsonl \
  --output data/verl/grpo_train_v1.parquet \
  --base-url http://127.0.0.1:5700
```

4. 在 veRL 环境中设置 `PYTHONPATH=/path/to/shopping-grpo-longhorizon/src`，使用 `configs/verl/shop_tools.json` 与 `configs/verl/shop_interaction.json` 做一次 2–4 条任务的租约/reward smoke。训练配置必须把 agent 的 `max_assistant_turns` 也设为 35，防止连续非法 tool call 绕过“执行步数”上限；通过后才填写其余超参数并启动 Vanilla GRPO。

## 仍待服务器验证

- 当前 veRL 版本能否正确解析 Qwen3.5 工具调用和加载 Qwen3.5 多模态 checkpoint；
- 组内 reward 方差、全 0 group 比例与单 GPU 可承受的 rollout 并发；
- 1000-task train split 的最终 probe 分布；
- 合并 checkpoint 后的固定 benchmark v2_50 复测，应与 adapter 推理一致。
