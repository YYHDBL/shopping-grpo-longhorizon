# 实验 02 模板：LoRA SFT 冷启动

> 状态：待运行。复制此文件为带日期的实验记录后再填写；不要在模板中填写结果。

## 问题与假设

- **问题：** 在固定 ShopSimulator benchmark 上，SFT 是否能让 Instruct 基座更稳定地完成多步购物工具调用？
- **假设：** 只对 assistant 文本和 tool call 计算 loss 的 LoRA SFT，会提高严格成功率并减少非法工具调用。

## 不变量

- benchmark：`data/benchmarks/shop_benchmark_v1.jsonl`；不得用于训练。
- 推理协议：同一 system prompt、12 个工具 schema、temperature=0、max_steps=35、max_tokens=512。
- 对照：同一个 Instruct 基座的 Base 评测结果。

## 本次运行卡片（由执行人填写）

| 字段 | 实际值 |
|---|---|
| 日期 / 操作人 | 待填写 |
| Git commit | 待填写 |
| 基座模型及 revision | 待填写 |
| 训练数据路径、SHA、有效行数 | 待填写 |
| chat template / max_length | 待填写 |
| LoRA target modules、rank、alpha、dropout | 待填写 |
| batch size、梯度累积、学习率、epoch | 待填写 |
| GPU、CUDA、峰值显存、训练耗时 | 待填写 |
| checkpoint / adapter 输出路径 | 待填写 |

## 运行命令与数据预检

记录完整的、不含密钥的命令。必须先贴 `inspect_sft_data.py` 的摘要，确认过长样本被显式丢弃、assistant label 不包含 user/tool observation，之后才训练。

```bash
# 待填写：数据预检命令
# 待填写：训练命令
# 待填写：adapter 加载与 benchmark 命令
```

## 结果

| 指标 | Base | SFT | 差值 |
|---|---:|---:|---:|
| 严格成功率 | 待填写 | 待填写 | 待填写 |
| 环境完成率 | 待填写 | 待填写 | 待填写 |
| r_type / r_att / r_option / r_price | 待填写 | 待填写 | 待填写 |
| 平均工具步数 | 待填写 | 待填写 | 待填写 |
| 非法动作/守卫拒绝 | 待填写 | 待填写 | 待填写 |

## 失败样例与判断

- 挑选成功、失败各至少 3 条 trajectory，链接 raw 输出并描述工具序列。
- 区分训练过拟合、模板/解析错误、环境故障和真实购物决策错误。
- 写下本次决定：继续扩数据、调训练配置、修数据质量，或停止进入 GRPO 前先修 SFT。
