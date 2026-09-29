# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.9.1` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
```

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train set: `data/grpo/train.parquet` (1,000 tasks)
- Validation set: `data/grpo/validation.parquet` (50 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v3

Hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh --dry-run
```

Train:

```bash
bash scripts/grpo.sh
```

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Maximum training steps | 500 |
| Save / validation frequency | 50 / 50 |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement | enabled (logging only) |

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
Advanced overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The reported comparison uses step 100. Select checkpoints using validation
metrics rather than assuming that the final training step is best.

## NCCL Duplicate GPU detected 排查

- **原因**：Ray 的 `worker_process_setup_hook` 导入 veRL 时触发 CUDA
  可用性查询，提前缓存设备映射。随后 Ray 设置各 worker 的
  `CUDA_VISIBLE_DEVICES`，CUDA 仍沿用此前的映射，导致多个 worker 使用同一张 GPU。
- **定位**：独立四卡 NCCL 通信检查通过；启用项目 hook 后，Ray 分配的 GPU UUID
  各不相同，worker 实际查询到的 GPU UUID 却全部相同。
- **修复**：在 `compat.py` 的 hook 导入 veRL 前设置
  `PYTORCH_NVML_BASED_CUDA_CHECK=1`，通过 NVML 检查设备可用性；hook 中不执行
  `torch.cuda.set_device()`，GPU 绑定在 Ray 完成资源分配后进行。
- **验证**：现有 Docker、535 驱动及 CUDA 13 兼容库环境下，四卡 NCCL 通信、
  veRL 通信初始化、小模型 FSDP 前向、反向和优化器更新全部通过，更新后的参数与
  单进程参考计算一致。验证脚本为
  [`tests/check_ray_fsdp.py`](../tests/check_ray_fsdp.py)。完整 GRPO 尚未验证。

运行检查时，通过 GPU UUID 显式限定 `CUDA_VISIBLE_DEVICES`，仅选择分配给当前任务的
GPU，避免占用其他服务使用的设备。
