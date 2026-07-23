# 实验 06：veRL 0.8 Vanilla GRPO 的有限 reward-group 动态采样

> 状态：CPU 实现与可复现补丁已完成，GPU signal smoke 待运行｜日期：2026-07-23

## 问题

SFT v2 merged 模型已经能在 ShopSimulator 中购买，但 `n=4` 的某个训练 group 可能四条
reward 完全相同。Vanilla GRPO 对同一 prompt 做组内归一化；全相同 reward 会产生全零
advantage，随后 `actor/loss=0`、`actor/grad_norm=0`。继续把这种 group 交给 optimizer
只会完成一次没有学习信号的假更新。

安装在 `.venv-grpo-v080` 的 `verl==0.8.0` 是固定运行时。实际审计
`verl/trainer/ppo/ray_trainer.py` 后确认，这个 wheel 的标准
`main_ppo → RayPPOTrainer.fit` 路径完全没有 `filter_groups`、
`max_num_gen_batches` 或 `seq_reward` 的执行逻辑。即使其他 veRL 分支或配置类出现过
类似字段，只在 YAML 中增加字段也不会改变这个 trainer 的控制流。

固定的原始文件为：

```text
.venv-grpo-v080/lib/python3.12/site-packages/verl/trainer/ppo/ray_trainer.py
SHA256 de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66
```

原始 `RayPPOTrainer.fit` 的关键位置是：

| 阶段 | 原始行号 |
|---|---:|
| `generate_sequences` | 1470 |
| `sleep_replicas` | 1471 |
| reward block / `extract_reward` | 1518 / 1525 |
| rollout-log-prob bypass | 1534–1536 |
| fallback `_compute_old_log_prob` | 1543 |
| Reference | 1579 |
| GRPO advantage | 1625 |
| `update_actor` | 1649 |

## 为什么不复制 DAPO Trainer

本实验仍然是 Vanilla GRPO：

- 环境终局 reward 不变；
- GRPO advantage 与 PPO clip loss 不变；
- Reference KL 不变；
- vLLM rollout log-prob 和 bypass mode 不变；
- Qwen3.5 LoRA actor、ShopSimulator AgentLoop 与环境租约不变。

DAPO Trainer 还包含 loss、clip、长度处理等与本问题无关的算法和控制流。复制整个
trainer 会扩大代码面和实验变量，也会让后续 veRL 版本审计困难。因此这里只给当前
固定 wheel 增加一个窄补丁：在 reward 已经产生、Reference 和 advantage 尚未计算的
位置过滤 group。

## 最小实现

项目侧纯函数位于 `src/shopping_grpo/verl_dynamic_sampling.py`。它不导入 Torch、Ray
或 veRL，只根据 `uid` 和每条轨迹的 `seq_reward` 返回：

- 应保留的 trajectory indices；
- 每个 uid 的四个 reward；
- 保留和丢弃的 group 统计。

判断使用绝对容差 `1e-8`。`max(reward)-min(reward) <= tolerance` 的 group 被丢弃；
存在真实分差的 group 才保留。`DataProto.select_idxs` 对同一批次的全部 tensor 和
non-tensor 字段使用同一组 indices，因此 `uid`、responses、mask、rollout log-prob、
reward extra info、task id、attention mask 和 position ids 保持对齐。

补丁控制流为：

```text
dataloader batch
  → AgentLoop/vLLM generate
  → ShopSimulator terminal reward
  → seq_reward 按 uid 分组
  → 常量 reward group 丢弃
  → 有分差 group 暂存
  → 不足 train_batch_size：rollout replicas 保持唤醒，取下一批
  → 凑够：合并并重新 balance
  → sleep rollout replicas
  → rollout-log-prob bypass
  → Reference
  → GRPO advantage
  → update_actor
  → 此时才提交 global_step
```

补采上限由项目顶层配置控制：

```yaml
shopping_dynamic_sampling:
  enable: false
  metric: seq_reward
  max_num_gen_batches: 3
  reward_tolerance: 1.0e-8
```

达到 3 批仍不足两个有效 prompt 时直接抛错，不计算 Reference、advantage 或 actor
update，也不拿常量 reward group 做假更新。当前每批为
`2 prompt × 4 rollout = 8 trajectory`，所以一次 signal smoke 最多生成
`3 × 8 = 24` 条轨迹。

## 可复现补丁与门禁

仓库保存：

```text
patches/verl-0.8.0-shopping-dynamic-sampling.patch
scripts/apply_verl_dynamic_sampling_patch.py
```

应用脚本严格检查 veRL 版本、`verl.__file__` 来源和原始 SHA256；应用前在原文件同目录
创建备份。已经应用时幂等成功，未知 hash 拒绝修改，`--restore` 只接受 hash 正确的
备份。没有网络访问，也不安装新依赖。

```bash
source .venv-grpo-v080/bin/activate
PYTHONPATH=src python scripts/apply_verl_dynamic_sampling_patch.py
PYTHONPATH=src python scripts/apply_verl_dynamic_sampling_patch.py --check
PYTHONPATH=src python scripts/apply_verl_dynamic_sampling_patch.py --restore
```

`scripts/check_grpo_runtime.py` 会解析与正式入口完全相同的 Hydra overrides。动态采样
启用时，它在加载模型前拒绝以下任一情况：

- veRL 不是 0.8.0；
- 补丁 marker 缺失；
- 项目纯函数不可导入；
- bypass mode 不是 true；
- rollout log-prob 未开启；
- `max_num_gen_batches` 不是正数。

## 后续 signal smoke

本次提交不运行 GPU。后续只用临时的 task 21934 两行 parquet 执行一次：

```bash
export GRPO_MODEL_PATH=/root/autodl-tmp/shopping-grpo-longhorizon/checkpoints/qwen35-2b-shopping-sft-v2-merged
export GRPO_TRAIN_FILE=/root/autodl-tmp/shopping-grpo-longhorizon/outputs/grpo_signal_smoke/task_21934_twice.parquet
# 本次 validation 完全关闭；复用临时 train parquet 只用于通过入口的文件存在性检查。
export GRPO_VAL_FILE="$GRPO_TRAIN_FILE"
export GRPO_OUTPUT_DIR=/root/autodl-tmp/shopping-grpo-longhorizon/outputs/grpo_signal_smoke/checkpoints

bash scripts/run_vanilla_grpo.sh \
  shopping_dynamic_sampling.enable=true \
  shopping_dynamic_sampling.metric=seq_reward \
  shopping_dynamic_sampling.max_num_gen_batches=3 \
  shopping_dynamic_sampling.reward_tolerance=1e-8 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1
```

验收时以 `SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH` 和
`SHOPPING_GRPO_DYNAMIC_SAMPLING_READY` 结构化日志还原每批 group reward、过滤结果、
最终训练 group 和总轨迹数。只有 `actor/loss != 0`、`actor/grad_norm > 0` 且
`global_step=1` 才能证明非零学习信号。

## 已知边界

1. 补丁固定绑定 veRL 0.8.0 当前 wheel 的原始 hash；重装或升级后必须重新审计，脚本
   会拒绝未知文件。
2. 多批有效 group 通过 `DataProto.select_idxs` 与 `DataProto.concat` 合并。CPU 测试能
   验证选择语义、补丁与配置，真实 Ray/FSDP/vLLM 生命周期仍需后续单步 GPU smoke。
3. 当一批产生的有效 group 超过当前剩余容量时，只接收最先出现、足以凑满
   `train_batch_size` 的 group；其余已生成轨迹不会进入该次更新。
4. 动态模式仅支持 `metric=seq_reward`、Vanilla GRPO、rollout-log-prob bypass。它不是
   通用 veRL Dynamic Sampling 或 DAPO 实现。
