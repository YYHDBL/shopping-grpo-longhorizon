# 实验 03 结果：SFT v2 — liger+SDPA+action-only 完整训练

> 状态：训练完成，eval 待跑｜日期：2026-07-21

## 配置

| 参数 | 值 |
|---|---|
| max_length | 20480 |
| 技术栈 | liger-kernel + SDPA + action-only 数据 |
| 训练样本 | 331/361 (92%) |
| 验证样本 | 18/19 (95%) |
| epochs | 3 |
| LoRA | r=16, alpha=32 |
| 量化 | 无（BF16 裸跑） |

## 结果

| 指标 | v1 | **v2** |
|---|---|---|
| train_loss | 0.554 | **0.084** |
| eval_loss epoch1 | 0.601 | **0.082** |
| eval_loss epoch2 | 0.578 | **0.083** |
| 峰值显存 | 39.4 GiB | 43.0 GiB |
| 耗时 | 68 min | 145 min |

eval_loss 下降幅度：v1 的 1/7。action-only + 更大数据量效果显著。epoch 2→3 仍趋平，现有数据已被充分学习。

## vs v1 关键差异

1. 数据利用率 59% → 92%（多 117 条样本）
2. 上下文长度 12K → 20K（保留复杂购买轨迹）
3. 数据格式 Full-CoT → action-only（去掉思考，专注工具调用）
4. 损失函数 标准 CE → liger fused（显存省 40%）

## 下一步

用固定 benchmark 跑 Base vs SFT v2 对比。产物路径：
- Adapter: `outputs/sft_runs/qwen35_2b_lora_v2/adapter/`
- 实验分析: `docs/experiments/03-sft-v2-memory-and-action-only-2026-07-20.md`
