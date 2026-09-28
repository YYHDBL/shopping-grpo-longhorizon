# V2.0 方案讨论交接

> 日期：2026-09-23。
>
> 用途：交给下一位 Agent 继续与用户讨论并对齐 V2.0 方案。当前只讨论设计，不实施代码、训练或正式评测。

## 1. 当前工作区

- 仓库：`/data/jyh-yyh/shopping-grpo-longhorizon`。
- 分支：`V2.0`，基于当时最新 `origin/main`。
- 当前代码和配置未修改。
- 设计基线：`docs/plans/2026-09-23-v2-design-baseline.md`。
- V1 问题 → V2 修改对照（更新说明）：`docs/plans/2026-09-23-v2-update-notes.md`。
- 本交接文档和设计基线当前均未提交；不要删除或覆盖用户已有工作。

下一位 Agent 必须先完整阅读设计基线，再继续提问。不要重新从零研究或反复询问已经确认的事项。

## 2. 项目目标和硬边界

- 从头重新检查环境、任务数据、Teacher 轨迹、失败分析、SFT、在线 GRPO、rollout、工具与 observation、Judge/Rubric 和最终分析。
- 目标是规模较小但流程完整、可复现、可观测、能形成 bad-case 数据飞轮的开源购物 Agent 项目。
- 不做多轮用户对话；只做单轮购物需求下的多步工具交互。
- 固定 `vLLM==0.25.1`、`veRL==0.8.0`，未经验证不直接改版本。
- 模型为后训练版 `Qwen/Qwen3.5-9B`，双 A800 80GB、BF16、全参数微调，FSDP 为基线。
- 用户可使用 GLM Coding API，暂时不用担心 Teacher 调用预算。
- 未得到用户明确指示前，不修改实现代码。

## 3. 已确认的数据事实

- 主数据：`environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz`。
- 总计 23,421 条；训练 21,962 条；评测 1,459 条。
- 非空画像 4,666 条；评测集中 1,343 条有画像、116 条无画像。
- Attribute 数量均值 4.53、中位数 4。
- 论文的 Options Count 是商品规格组合数，不是 `instruction_options` 长度；全量均值 17.12、中位数 7。
- 当前历史 Final-200 不作为 V2.0 正式最终评测集。

## 4. 已确认的数据设计

### Query 与 Gold

- 以现有 21,962 条训练 Query 为主体，不全量重生成。
- 只对坏数据和分布缺口定向修复或补充。
- Query 是用户需求，Gold 是参考答案；两者冲突时不能为了 Gold 改写 Query。
- 训练任务优先重配可靠 Gold，找不到则不进入训练。
- 评测任务不静默删除；无效或不可验证项保留并单独报告。
- 原始 `attributes` 和 `instruction_options` 标注直接信任，不做全量重新提取或模型复核；明确 bad case 后续单独回流。

### 难度

使用二维规则：

```text
easy:   attributes < 4  AND option_combinations <= 7
hard:   attributes >= 9 OR  option_combinations >= 21
medium: 其余
```

分布：

| Split | Easy | Medium | Hard |
|---|---:|---:|---:|
| Train 21,962 | 3,950（18.0%） | 12,908（58.8%） | 5,104（23.2%） |
| Eval 1,459 | 206（14.1%） | 788（54.0%） | 465（31.9%） |

- domain/category 只做二级覆盖，不进入难度公式。
- 训练尽量保持源分布近似；最终评测保持源 eval 自然分布。
- 第一版不额外合成无解任务。

### 个性化

- 不复现论文的个性化 benchmark。
- 只保留 `no_profile / aligned / irrelevant`，比例为 `50% / 25% / 25%`，三类都进入训练和评测。
- 每个基础 Query 只分配一种条件，不复制三份；按 difficulty、domain、category 分层分配。
- 当前明确请求优先，画像只是软偏好。
- 训练集约 `10,981 / 5,491 / 5,490`；评测集约 `730 / 365 / 364`。

## 5. 已确认的 Reward 方向

```text
Gold ASIN
  -> 本地代码直接满分

明确硬错误
  -> 本地代码直接判错

非 Gold 且未触发硬错误
  -> OpenRouter Jev 判断 alternative
```

- 任务必须先通过离线数据验收，Gold 才能按 ASIN 直接满分。
- 完全满足需求的 alternative 与 Gold 同分。
- Jev 固定使用 `typesafe/jev-1.13`，不使用 latest 别名。
- 每个候选只问一个 Choice：`fully_satisfies / partially_satisfies / does_not_satisfy / insufficient_evidence`。
- Jev 只接收实际用户请求、可选画像、真实候选商品字段、实际 options 和成交价格。
- 不向 Jev 提供 Gold 商品、完整轨迹或原始 DOM。
- 字段缺失不得合成；证据不足、API 超时或失败都标记 Reward 不可验证，不能当作模型失败记零。
- 在线 GRPO 先本地过滤，再并发调用 Jev，并缓存相同判断；具体概率阈值和部分奖励尚未冻结。

## 6. 下一步需要和用户对齐的问题

按下面优先级逐项讨论，一次只问一个真正需要用户拍板的问题；先给清晰推荐和理由，不要一次抛出整套问卷。

1. ~~画像构造~~ **已确认并执行完毕**（方案 2026-09-23，执行 2026-09-25~26）：三条件分配 + 画像池（4,666 份，773 泄漏标记）+ 配对复核 9,554 条（agent 分片 4,961 + Jev 4,595）+ 重配循环 3 轮。最终 train 18,014 条：no_profile 9,336 / aligned 4,267 / irrelevant 4,411（52/24/24，353 条三轮重配失败降级 no_profile）；eval 1,092 条：569/264/259。画像渲染统一剔除搜索关键词等行为清单字段（`src/shopping_grpo/persona/render.py`）。最终文件 `outputs/split/tasks_final.jsonl`（含难度、条件、画像引用、train/dev 归属，dev 精确 1,050）。
   执行教训（2026-09-25 用户指示）：大批量语义判断优先用 Jev（API 稳定、分钟级、$0.1 量级），subagent 并行易触发账户限流且耗时不可控。
2. ~~Query–Gold 验收~~ **已确认并执行完毕**（2026-09-23 确认方案，2026-09-24~25 全量执行定案）：两级验收 + 六分片复查 + 异常定价剔除。最终 accepted 19,106 / semantic_fail 3,310 / hard_fail 954 / bad_pricing 39 / unverifiable 12；Jev criteria 升 v2（价格容差 + 同义表述）；定案文件 `outputs/acceptance/final_verdicts.jsonl`。详见设计基线 4.2。
3. ~~开发集~~ **已确认**（2026-09-23）：约 1,050 条（5%），difficulty × domain × 画像条件分层，category 只做覆盖报告；dev 不进 SFT/GRPO，仅用于训练中评估、选 checkpoint、迭代 Rubric。
4. ~~定向补充数据~~ **已确认**（2026-09-23）：第一版不补、不预设流程；验收后如实报告损失，损失大到影响训练/分布时再单独讨论。
5. ~~Teacher 轨迹~~ **已确认并采集完成**（方案 2026-09-23，执行 2026-09-25~27）：双 Teacher 分治（GLM-5.3-Flash 与 DeepSeek V4.1 Flash 各半分片、各 8 并发、环境 16 槽），每题预算 3 次成功即停，提示词分层（Teacher 版含决策规则，Student 版仅协议层）。最终 6,579 题处理，教材 5,852 条（GLM 2,991 / DS 2,861，通过率 89%），hard 占 34.6%，平均 5.9 步；teacher-hard 队列 727 题。输入 4.78 亿 + 缓存读 5.33 亿 + 输出 871 万 token。全轨迹留存 `outputs/collection/trajectories/`。执行期事故与修复：SSH 隧道代理断连两次烧队列（已改为客户端代码级直连 + 全局熔断，彻底不依赖代理）；tmux 内 Ctrl+C 会杀采集进程（退出 tmux 用 Ctrl+B 再 D）。
6. ~~SFT 数据契约~~ **已确认**（2026-09-23）：全链路关思考；loss 只算模型行动 token，observation/prompt 全 mask；超长轨迹丢弃并监控丢弃率（>5% 调上限）。
7. **Harness 与可观测性** — 埋点/追踪等工程细节 Agent 给默认（基线 6 已覆盖）；但用户提供了一组 Harness issues（见 8.5 节：tool_choice 强制、详情页低价值工具、observation 截断），**尚未讨论，后续 Harness 讨论时必须回应**。
8. ~~正式评测与错误归因~~ **已确认并完成校准**（2026-09-23 定方案；2026-09-27 校准执行完毕）：50 条考卷 + DeepSeek 盲判裁决（一致率 88% 过线、满分精确率 93%），据此冻结 Jev v3（价格低位不算违规、字段沉默不单独扣死）。考卷与结果在 `outputs/calibration/`。
9. **在线 GRPO** — Reward 数值映射 **已确认**（2026-09-23）：v3 数值为基础，仅改两处——替代品/Gold 同为 1.0，Jev partial 固定 0.25；买错 -0.85 最重；不可验证从组内剔除。**RL 任务池已确认**：与 SFT 任务隔离，优先用未用过的约 12,600 道。**对照设计已确认**：三档主线（零样本 / SFT-only / SFT+GRPO），不做多 RL 算法对比。**仍待定：group size（默认 8，工程项）、Jev 概率阈值（校准后）、batch/序列长度/offload（实测项）。**

## 8.5 Harness 待解决议题（用户提供 issue 材料，2026-09-23）

**议题 1 已确认（2026-09-23，三件套）**：
- 环境回合 `tool_choice=required`：每回合必须出工具调用，取消"纯文本交卷"出口；结束只有 `buy_now` 或 `finish_without_purchase` 两个正路——结束是动作，不是独白。
- 移除 `think` 工具（与全链路关思考一致，且 required 下它会成为被迫乱调的垃圾出口）。
- "文本 + 工具调用"混合回复不拒绝：文本保留进上下文，工具照常执行；单回合只执行第一个工具调用、多余丢弃并记录的现行规矩不变。
- vLLM 0.25.1 / veRL 0.8.0 对 required 的支持列入兼容性冒烟清单。
- 备选方案（auto + 纯文本回敬重试）未被采用。

后续讨论 Harness 时必须纳入考量，来源为针对 2B 模型 strict 成功率问题的分析（调整 Harness 配置后 2B base 达到约 50% strict accuracy）：

1. ~~工具调用不强制~~ **已确认**（见上，三件套）。
2. ~~详情页工具信息增益低~~ **已确认**（2026-09-23 定为删 reviews，2026-09-24 用户修订：环境与工具集完全不动）：
   - `view_reviews` 与四个子页工具全部保留，ShopSimulator 环境不做任何修改（Environment v2.1 固定契约）。评论页在当前数据下内容恒空，允许模型最多浪费一步。
   - 同一商品同一子页（含评论页）只允许访问一次：第二次调用由动作守卫直接拦截（不消耗环境步骤，回敬提示"信息已在上下文中"），复用现有 blocked 机制。
   - 不做过程奖励（拒绝"以新增证据计 progress"的 shaping 思路，维持终止型 Reward）；但轨迹记录子页访问占比、重复拦截次数，进入错误归因副表。
3. ~~Observation 截断过狠~~ **已确认**（2026-09-23）：预算按实测 P95 定（目标 95% 页面不截断），试跑临时值 search 3072 / detail 6144 / generic 1024（约翻倍）；截断按字段价值排序（动作信息硬保不动，正文先砍自由文本和长标题，key_attributes / options 最后砍）；两个监控闭环——截断分桶成功率对比、与 SFT 超长丢弃率（>5% 报警）联动配平预算与序列长度。

与 V2.0 已确认设计的关联：三议题都直接影响 Teacher 采集质量与 SFT 教材（截断太狠 = 学霸看到的信息也残缺），Harness 讨论时需先于 GRPO 参数定案。

## 7. 建议下一问

直接从画像构造开始。建议向用户确认的核心方案是：

> `aligned` 是否优先复用并清理现有画像，`irrelevant` 是否通过跨领域重配现有画像生成，同时保持 no-profile/aligned/irrelevant 三组在 difficulty、domain、category 上近似同分布？

不要先讨论字段级规则、代码结构或采样脚本，除非用户明确要求。

## 7.5 执行顺序（2026-09-24 确认：数据链路先行，冒烟不提前）

按依赖关系排序，每步的产出是下一步的输入：

1. **数据验收管线**：两级全量验收（代码硬约束 + Jev-1.13 语义），产出验收报告（损失率、invalid/unverifiable、换 Gold 标记）；跑批时顺手记录各页面类型 observation 原始 token 分布 → 为 P95 预算提供数据。
2. **画像构造**：泄漏标签、aligned 清洗 + 池内匹配补齐、irrelevant 跨领域重配 + 语义复核（GLM）；difficulty 标注；dev 1,050 分层划分。
3. **Harness 修改**：tool_choice=required、删 think、删 view_reviews、子页重复拦截、截断字段分级（预算用第 1 步的实测分布定）。
4. **Teacher 小规模试点**：目的不是出教材，是产出灰区替代商品候选 + observation 预算验证。
5. **Jev 校准冻结**：Gold 场景 150 条（第 1 步顺手抽）+ 灰区场景 150 条（第 4 步产出）→ GLM 预标 → 用户裁决约 100 条 → 冻结版本/提示词/阈值。
6. **Teacher 正式采集**：6,000 × 4，用冻结后的 Jev 和修改后的 harness。
7. **SFT 前冒烟**：vLLM 0.25.1 × veRL 0.8.0 兼容性（含 required、agent loop token 边界、loss mask 对齐验证）——**在 SFT 启动前做，不提前**。
8. **SFT → GRPO → 三档评测**。

开源交付范围（license、权重/数据发布）：出结果后再定（用户 2026-09-24 确认暂不讨论）。

## 8. 沟通方式

- 使用中文和大白话。
- 用户不喜欢过度设计，也不希望被细枝末节问题打断。
- 只问必须由用户决定的产品/研究选择；普通工程细节由 Agent 给出合理默认。
- 对当前实现保持怀疑，但不能为了“更完整”而推翻已确认的简单方案。
- 每次确认后更新设计基线；未确认内容必须明确标为待定。
