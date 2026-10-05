# TRACE Phase 0 离线验证报告（2026-10-05）

## 结论

**不宜直接进入 Phase 1（训练）**。核心原因：TRACE 的 proxy 在本项目上
**系统性惩罚核验行为**（view_* 工具全负 credit），与"修复探索不足"的目标
方向相反；且训练成本翻倍。但验证过程产出了对"过程奖励"路线的通用认知。

## 方法

- 算法：main 分支 `trace.py`（282 行，TRACE 论文 arXiv 2607.13988）已搬入 V2.0
- gold target：环境 `include_trace_target` 直接返回，1092 个全部落盘
- 评分：frozen ref = sft-merged 单卡，轨迹=run3-50 评测集按结果分层抽 163 条
- 脚本：`scripts/trace_phase0_score.py` + `trace_phase0_analyze.py`

## 问题 ①：探索动作是否得正 credit？——**方向性分裂**

δ（相邻 turn 的 gold log-prob 变化，纯探索信号）：

| 动作 | δ均值 | δ>0 比例 | 判读 |
|---|---:|---:|---|
| search_products | **+0.388** | 76% | ✅ 强正（搜对方向）|
| prev_page / back_to_search | +0.06 | 91~96% | ✅ 正（回列表）|
| open_product | +0.103 | 34% | ⚠️ 双峰（开对正/开错负）|
| **view_description** | **-0.036** | 12% | ❌ 负 |
| **view_features** | **-0.082** | **3%** | ❌ **强负（97% 为负）** |
| view_reviews | -0.025 | 26% | ❌ 负 |
| select_option | -0.006 | 48% | ➖ 中性 |
| buy_now | +0.077 | 94% | ✅ 正 |

**机制**：teacher-forcing 打分中，优先信息页（长自然语言）进入 prefix 会
**提高**预测格式化 target（"最终应购买商品：{json}"）的难度 → log-prob 下降。
**"看得越细"越像"偏离目标"**——而核验正是我们要鼓励的探索行为。

## 问题 ②：替代品冲突——**实锤**

V(S_T)-V(S_0)（gold 可定位性的全程变化）按结果分组：

| 结果 | V 跨度 | 尾部 credit |
|---|---:|---:|
| gold_purchase | +2.07 | +1.64 |
| valid_alternative | +1.96 | +0.93 |
| wrong_purchase | +1.20 | -1.57 |
| repeat_loop | +1.07 | -1.04 |
| **partial_alternative** | **+0.94** | +0.42 |
| max_steps | +0.87 | -1.04 |

- partial（reward 认可的替代品，0.25 分）轨迹的 turn credit 只有 gold 轨迹的
  **45%**——proxy 与结果定义部分矛盾（研究文档此前预判的风险，实测确认）
- proxy 整体有判别力但度不大：成功 +2.07 vs 非成功 +1.24（差 0.83）

## 问题 ③：成本——**翻倍**

- 163 条（2443 次前向、1691 万 prefix tokens）耗时 54.7 分钟（串行）
- 单条 ~20s → 训练外推 256 轨迹/步 ≈ **+85 分钟/步**（步时 60 → 145 分钟）
- 论文式批处理乐观优化后：估计仍 +30~60 分钟/步

## 通用认知（可复用到其他"过程奖励"设计）

1. **"gold answer log-prob"类 proxy 对长文本 observation 天然敏感**——任何
   基于 teacher-forcing 的稠密信号都要先测它是否惩罚核验行为
2. TRACE 的干净信号在**搜索**（+0.388）——"搜对方向"被准确捕捉；但购物
   Agent 的探索不只是搜索，还包括比较与核验，后者被 proxy 判负
3. 若未来再走过程奖励路线：proxy 需以"核验后的正确判断"而非"gold 字符串
   可预测性"为目标（如 structured-goal value estimator——研究文档亦指向此）

## 遗留资产

- 算法：`src/shopping_grpo/training/grpo/trace.py`（可复用其 turn 切分与评分管道）
- target 表：`outputs/trace_phase0/targets.jsonl`（1092 条）
- 评分/分析脚本：`scripts/trace_phase0_score.py`、`trace_phase0_analyze.py`
- 数据：`outputs/trace_phase0/credits_run3_50.jsonl`（163 条 turn-level credit）
