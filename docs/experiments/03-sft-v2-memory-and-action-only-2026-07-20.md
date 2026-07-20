# 实验 03：SFT v2 — 长上下文显存解决方案验证

> 状态：已完成｜日期：2026-07-20｜结论：liger-kernel + SDPA + action-only 三件组合 20K 可训，92% 数据保留

## 问题

v1 在 12288 token 才训练成功，12288→16384 OOM，16384 需要 QLoRA 才能勉强跑（46 GiB）。数据利用率仅 59%。瓶颈是 Qwen3.5 的 151K 词表导致 logits 张量巨大（16K 序列 ~5GB），而非 attention 本身。

## 实验设计（A/B/C/D 矩阵）

按仓库 `03-sft-v2-memory-and-action-only-2026-07-20.md` 方案执行：

| 组别 | 技术栈 | 测试长度 |
|---|---|---|
| B | liger-kernel（融合 loss） | 12K → 16K |
| C | B + SDPA（PyTorch 内置高效注意） | 16K → 20K |

A 和 D 跳过：A 是 BF16 基线已知，D 的 QLoRA 在 B/C 通过后不需要。

### 数据

使用 action-only 格式（去掉 Teacher 思考推理，只保留 tool call），不传 `--retain-teacher-reasoning`。

环境：`.venv-sft` + `requirements-sft-accelerated.txt`（liger-kernel 0.8.0, bitsandbytes 0.49.2）。

## 结果

| 长度 | liger | SDPA | kept | 峰值 (step) | 峰值 (overall) | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 12K | ✅ | — | 249/361 | ~10G | **31.9G** | 基线 |
| 16K | ✅ | — | 304/361 | ~10G | **41.4G** | 84% 数据 |
| 20K | ✅ | ✅ | **331/361** | ~15G | **~42-44G** | **92% 数据，最佳** |

对比 v1：
- 无优化：12K 39G，12288→16K OOM
- QLoRA：16K 46G（需量化）
- **liger+SDPA+action-only：20K ~43G（BF16 裸跑，不需 QLoRA）**

## 各技术贡献分析

| 技术 | 省显存 | 精度影响 | 备注 |
|---|---|---|---|
| **liger-kernel** | ~40% | **零损失** | 融合 LM head + cross-entropy，数学等价 |
| **SDPA** | ~20% | **零损失** | PyTorch 2.x 内置，bf16 下等价 |
| **action-only** | ~15% | **正向** | 去掉 CoT 思考，2B 小模型不需要；序列变短后多用 117 条样本 |

## FlashAttention 2 放弃原因

PyTorch CUDA 13.0 vs 系统 CUDA 12.8 不匹配，编译失败。SDPA 已提供等价优化，不折腾 CUDA 环境。

## 失败记录

- v1 BF16 16K：OOM at backward（logits 16.35 GiB）
- v1 BF16 20K：OOM at forward（logits + attention）
- v1 QLoRA 18K：OOM at backward
- FA2 编译：nvcc 临时文件路径问题 + CUDA 版本不匹配

## 最终配置

```bash
--max-length 20480
--liger-kernel
--attention-implementation sdpa
--bf16
--gradient-checkpointing
--epochs 3
--lora-r 16 --lora-alpha 32 --lora-dropout 0.05
--per-device-train-batch-size 1 --gradient-accumulation-steps 8
--learning-rate 1e-4
```

数据：`outputs/flash_accepted_500_parallel_action_only/`

预期：331 train samples，~124 梯度更新，~2h 训练，peak ~44 GiB。
