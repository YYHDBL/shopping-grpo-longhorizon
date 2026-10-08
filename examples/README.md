# 数据样本示例

从项目数据中提取的少量真实样本，用于快速理解数据形态。全量数据见 [数据索引](../docs/data-index.md)。

## `sft_sample.json`

一条 SFT 教材轨迹（`outputs/sft_dataset/samples.jsonl` 中的一条）。

- 难度 hard，Teacher 为 DeepSeek，15 个工具步骤
- 动作序列：搜索 → 打开 → 选规格 → **看参数/看功能（核验）** → 返回搜索 → 换商品 → 再搜索 → 打开 → 选规格 → 购买
- 看点：完整的"搜索-比较-核验-购买"链路；`messages` 为 Student 版 prompt（仅协议层）；实际训练时按消息计算 loss mask（仅 assistant 动作参与训练）

## `flip_case_task18_*.json`

同一道题（task 18）上两个模型的评测轨迹对照——**"翻盘案例"**（SFT 失败、GRPO 成功）：

| 文件 | 模型 | 结果 |
|---|---|---|
| `flip_case_task18_sft.json` | SFT | 失败（买错/未达标） |
| `flip_case_task18_grpo.json` | GRPO run3-50 | gold_purchase 成功 |

- 两条轨迹步数相同（5 步），差异在**搜索词与候选选择**：
  GRPO 的搜索词加入了核心约束词（"软底"），打开的商品标题明确覆盖全部需求属性；
  SFT 的搜索词更宽泛，打开的商品缺少关键属性
- 这两条轨迹正是轨迹分析（`scripts/compare_trajectories.py` + `scripts/deep_read_l3.py`）
  的 41 个翻盘样本之一，逐题归因见
  [轨迹分析记录](../docs/plans/2026-10-03-trajectory-analysis-plan.md) 第八节
- 看点：**RL 的行为改变不是"更努力"而是"更精准"**——同样的步数、同样的流程，
  搜索词的质量不同

## `trajectories.jsonl`

早期（V1 时代）的示例文件，保留作历史参考。

## 轨迹文件的结构

评测/采集轨迹的关键字段：

```text
task_id         任务编号（对应 tasks_final.jsonl 的 record_id 索引）
messages        完整对话：system(协议) / user(需求) / assistant(工具调用) / tool(环境返回)
terminal_result 终局判定（reward_type 等）
status          done / error
metrics         过程指标（时延、token 用量、环境交互统计）
```
