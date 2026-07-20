# 实验 03 模板：Vanilla GRPO 接入与诊断

> 状态：待运行。当前不实现 PRM-Lite、LATA 或额外 Reward Model；先跑可解释的 Vanilla GRPO 基线。

## 问题与假设

- **问题：** 在完成 SFT 冷启动后，ShopSimulator 的终局严格 reward 能否继续提高购物成功率？
- **假设：** SFT 提供基本的工具调用策略，Vanilla GRPO 再通过环境交互改善类别、属性、规格和价格四项约束的满足率。

## 必须先确认的前置条件

1. 有可复现的 Base 与 SFT benchmark 结果，且使用固定 held-out task；
2. rollout、reward、环境释放和 checkpoint 导出都有 smoke test；
3. 明确训练 task 与 `shop_benchmark_v1` 完全隔离；
4. 记录每轮 group reward 的分布，防止把奖励饱和误判为收敛。

## 本次运行卡片（由执行人填写）

| 字段 | 实际值 |
|---|---|
| 日期 / 操作人 / Git commit | 待填写 |
| 起始 SFT adapter / 模型 revision | 待填写 |
| 训练 task 范围及与 test 的重叠检查 | 待填写 |
| veRL 版本、GPU、并行策略 | 待填写 |
| group size、rollout 数、最大步数/生成长度 | 待填写 |
| reward 定义与各分量记录方式 | 待填写 |
| 学习率、KL、batch、总 step | 待填写 |
| checkpoint 与日志路径 | 待填写 |

## 训练期间必须记录的曲线

- 严格 reward、四个 reward 分量、环境完成率；
- group 内 reward 的均值、方差、全 0 / 全 1 比例；
- 平均工具步数、生成 token 数、超时和环境错误数；
- 固定 benchmark 上的严格成功率，而不是只看训练 reward。

## 结果与诊断

| 指标 | Base | SFT | Vanilla GRPO | 结论 |
|---|---:|---:|---:|---|
| 严格成功率 | 待填写 | 待填写 | 待填写 | 待填写 |
| r_att / r_option | 待填写 | 待填写 | 待填写 | 待填写 |
| 平均步数 / 错误率 | 待填写 | 待填写 | 待填写 | 待填写 |

如果出现退化，不要立刻加复杂奖励。先判断：是任务泄漏、工具 parser、环境租约、SFT 冷启动不足、group reward 饱和，还是长轨迹信用分配问题。只有用日志证实 Vanilla GRPO 的具体瓶颈后，才讨论参考外部项目的过程奖励或长度归一化方案。
