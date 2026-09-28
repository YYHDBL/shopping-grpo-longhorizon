# 数据验收记录：问题、方法、结果

> 日期：2026-09-25。范围：`fine_items_eval_train_all.json.gz` 全量 23,421 条（train 21,962 / eval 1,459）。
> 定案文件：`outputs/acceptance/final_verdicts.jsonl`。后续数据工作一律以定案文件为准。

## 一、源数据发现的问题

1. **硬约束问题（954 条，4.1%）**：本地代码直接判死，无需语义判断。
   - 必选规格缺失或不可用 501 条：用户指令点名的规格值在商品选项里找不到，或已下架（is_available=false）；
   - 实价超预算 453 条：按目标规格组合的实际价格（环境 Reward 同款 `resolve_variant_price` 计算）超出指令预算上限。

2. **语义问题（Jev 初判 5,824 条，24.9%）**：商品字段与用户需求有出入。坏因分布（六个分片报告归纳一致）：
   - 价格失配约六成：Gold 价格远低于用户价位（需求 5400 元给 500 元）、定金链接的定金价被当作全款价、单件计价对多件需求；
   - 规格错位：尺码、容量、数量、颜色不符（M 码对 L 码需求、30ml 对 80ml、1 支对 2 支）；
   - 品类级错配 24 条：PS4 需求配 PS5 版、要风扇配水冷头、要礼服配拖尾配件。

3. **Jev 判定偏严**：与独立复查、人工盲判三方对照后确认。锚点对比（50 条盲判）显示 Jev 的 partial 判定约一半存疑；分歧约六成源于其判定标准没有价格浮动口径（用户说"40 元左右"给 42 元即判不符），约一成源于不认同义表述（"护颈椎"对"保护颈部脊柱"）。

4. **异常定价（39 条）**：Gold 标价 ≤1 元的链接（定金、意向金或异常标价），任何口径下都不可用。

## 二、方法

三段式：两级验收 → 六分片复查 → 定案合并。

1. **第一级（本地代码，零成本）**：结构完整性、必选规格核验、预算核验。价格用环境同款 `resolve_variant_price`，保证验收与 Reward 判分同一把尺子。
2. **第二级（Jev decisions API）**：对第一级未拦截的 22,467 条做语义四分类（fully / partial / does_not / insufficient）。实际版本 `typesafe/jev-1.13-20260917`，总成本 $0.70，仅 11 次网络重试。
3. **复查（六个轻量 subagent 并行分片盲判）**：对 5,824 条 semantic_fail 逐条独立重判，每分片约 970 条、每轮 15 条。锚点校验：与 glm-5.3 盲判一致率 72%，分歧双向，无系统性偏宽偏严。
4. **定案规则（用户 2026-09-25 裁决：信复查）**：
   - 复查判 fully 且 Gold 标价 >1 元 → accepted（救回）；
   - 复查判 fully 但标价 ≤1 元 → bad_pricing（剔除）；
   - 复查维持 partial / does_not / insufficient → semantic_fail（确认坏）；
   - 其余维持原判定（hard_fail / unverifiable）。
5. **Jev 判定标准修订 v2**（`jev-gold-acceptance-v2`）：instructions 写入价格容差口径（"以内"= ≤X、"左右"= ±10% 浮动、区间闭合）、同义表述视为满足、主观偏好不扣分。已验收数据不重跑（放宽口径下原 accepted 必然仍 accepted，复查已完成宽口径重判）；v2 用于后续换 Gold 确认、Teacher 验收与灰区判分。

## 三、结果

| 判定 | 条数 | 占比 | 去向 |
|---|---:|---:|---|
| accepted | 19,106 | 81.5% | 进入画像构造与切分 |
| semantic_fail | 3,310 | 14.1% | 进换 Gold 流程 |
| hard_fail | 954 | 4.1% | 进换 Gold 流程（规格缺失 / 超预算） |
| bad_pricing | 39 | 0.2% | 确认坏（异常定价） |
| unverifiable | 12 | 0.05% | 可重跑 |

按 split：train accepted 18,014（82.1%，扣 dev 后约 16,964 可用于训练）；eval accepted 1,092（74.8%，可评分量）。

难度维度分布均匀（partial 率 easy 25.0% / medium 24.4% / hard 25.8%），判定无难度偏向。

## 四、产物清单

- `outputs/acceptance/level1_checks.jsonl`：第一级逐条明细
- `outputs/acceptance/jev_verdicts.jsonl`：第二级逐条明细（含概率、成本、耗时）
- `outputs/review/result_1..6.jsonl`：复查逐条明细（含理由）
- `outputs/acceptance/final_verdicts.jsonl`：定案（record_id + 最终判定 + 依据来源）
- 代码：`src/shopping_grpo/acceptance/`（管线）、`scripts/run_acceptance.py`（两级执行）、`scripts/jev_review.py`（复查分片与校验）、`scripts/finalize_acceptance.py`（定案合并）
