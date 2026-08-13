# Evaluation：从一条 Benchmark Test 到最终比较

当前正式分母为 [Final-200 Clean](evaluation-dataset.md)。旧的
[Final-200 Benchmark Dashboard](evaluation-dashboard.html) 仅保留为历史归档。

本项目的正式评估不是“让一个模型看结果后打一个总分”，而是由代码硬检查、
DeepSeek V4 Flash Rubric 整理器、DeepSeek V4 Pro 轨迹 Judge 和最终聚合器组成。
四类结果始终分栏报告，不合成一个不可解释的总分。

> 文档中的 **Rubric** 指“逐任务评分标准”，不是向量检索式 RAG。它由代码候选和
> V4 Flash 共同生成，随后冻结并由 Baseline、SFT、GRPO 三个 Actor 共享。

## 1. 全流程

```mermaid
flowchart TD
    A["Final-200 Clean benchmark<br/>200 × task_id"] --> B["从 ShopSimulator 导出私有 TaskFacts<br/>Query + 目标商品结构化事实"]
    B --> C["代码提取 Rubric 候选<br/>category / brand / model / function / option / price"]
    C --> D["DeepSeek V4 Flash<br/>只能筛选、去重、描述和标注 hard/soft"]
    D --> E["Schema + candidate_id + Query span + hash 校验"]
    E --> F["每个 task 冻结一个 Rubric bundle<br/>三个 Actor 共用"]

    A --> G["Actor 在 ShopSimulator 执行一次 Rollout"]
    G --> H["规范化轨迹并生成稳定 event_id"]
    H --> I["代码硬检查<br/>Reward / terminal / legality / repetition / context / infrastructure"]
    I -->|infrastructure_invalid| J["not_judged<br/>仍保留在 200 题分母"]
    I -->|valid| K["构建 Judge-safe 输入<br/>移除 Reward、Gold 和 raw observation"]
    F --> K
    K --> L["DeepSeek V4 Pro<br/>逐 Rubric 判断 + 五维轨迹评分 + 错误分类"]
    L --> M["JSON Schema、event_id、rubric_id、模型与请求 Hash 校验"]
    J --> N["四面板结果拼装"]
    M --> N
    N --> O["固定 200 题分母汇总"]
    O --> P["按 task_id 配对比较<br/>Baseline ↔ SFT ↔ GRPO"]
```

Rubric 只需要为每个任务生成一次；它不依赖某个 Actor 的轨迹。三个模型随后在相同
任务上各自产生一条 Rollout，并独立交给 Pro Judge，避免 Judge 先看到其他模型的
结果而产生比较偏差。

## 2. Benchmark 中的一条 Test

正式 Final-200 Clean 文件只公开任务 ID，防止将盲测 Query 或目标商品意外送入训练流程：

```json
{"task_id": 8187}
```

开始评估后，代码按这个 ID 从冻结的 ShopSimulator goal 顺序中恢复私有 TaskFacts。
这条任务的 Query 是：

> 求一对卡通-永结同心款的高档酒红色木梳，希望配备礼盒，能作为新娘结婚的陪嫁
> 物品，价格在20元左右。

TaskFacts 还包含目标商品的 category、title、brand、pricing、attributes、
customization options，以及 Reward v3 已编译的结构化需求。它们用于生成候选约束，
但目标商品私有字段不会进入 Actor，也不会直接进入 Pro Judge。

Final-200 Clean 的约束如下：

- 200 个任务；
- 与 SFT、GRPO train/validation 和历史 benchmark 零重叠；
- SHA-256：
  `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5`；
- 不用于 Prompt 调优、Rubric/Judge 校准或 checkpoint 选择；
- 每个模型每题一次确定性 Rollout。

## 3. 第一位 LLM：V4 Flash 生成冻结 Rubric

### 3.1 代码先生成候选，不让 Flash 自由发挥

代码从 Query、instruction 标注、目标商品结构和 Reward features 提取一个“可能约束”
超集。候选类型包括：

| 类型 | 例子 | 底层表示 |
|---|---|---|
| `category` | 商品应为新娘配件 | `product.category in_category ...` |
| `brand` | 品牌应为某品牌 | `product.brand eq ...` |
| `model` | 型号应为某型号 | `product.model eq ...` |
| `core_function` | 应支持高档/陪嫁等要求 | `product.attributes contains ...` |
| `option` | 颜色或组合规格应选某值 | `purchase.options.* eq ...` |
| `budget_upper` | 价格不超过 100 元 | `purchase.price lte 100` |
| `price_range` | 价格在 80–100 元 | `purchase.price between ...` |
| `price_preference` | 价格在 20 元左右 | `purchase.price approximately 20` |

每个候选都有固定的 `candidate_id`、字段、操作符、期望值、hardness hint、Query
span、数据来源和 selection guidance。V4 Flash 无权创造新的底层字段、操作符或值。

示例 task 8187 的代码候选共有 7 条：品类、高档、结婚、陪嫁、卡通-永结同心、
“【卡通-永结同心】2个装”选项，以及“20 元左右”的价格偏好。

### 3.2 V4 Flash 的完整 System Prompt

当前冻结版本为 `rubric-curator-v1-draft-r4`，模型为
`deepseek-v4-flash`。下面是代码中的完整提示词：

```text
你是当前 Shopping Agent 项目的需求 Rubric 整理器，不是自由生成需求的助手。

你只能从输入的 candidates 中选择用户 Query 确实表达的约束，并做简短自然语言化。
严禁新增 candidate_id，严禁修改候选的底层字段、操作符或期望值，严禁把目标商品的
全部属性自动视为用户需求。

先保证完整覆盖 Query 中每一个彼此独立的明确要求，再做最小化和去重。“最小”只表示
同义或上下位重复要求不重复计分，绝不表示少选要求。每个 candidate 的
selection_guidance 是强制选择规则。

每条选择必须有 Query 原文直接支持：
- 泛化的目标商品属性不能仅凭常识或商品字段入选；
- 同一用户要求已经由更具体的 option、规格或价格候选覆盖时，不再拆成多个含义重叠
  的泛化 core_function；
- 品类词不能因为碰巧等于目标商品 brand 而被选成品牌约束；
- “适用于/兼容某品牌”是兼容性要求，不代表所购商品自身必须属于该品牌；
- Query 明确给出商品类型时，品类本身是一条独立要求，应和功能、规格分别保留；
- 组合 option 可以承载 Query 明确要求的多个规格，但不能借机加入用户未要求的品牌、
  型号、数量或实质规格；
- 每条入选约束必须提供 Query 中连续、逐字存在且能独立支持该约束的非空原文片段。

例：Query “推荐移动电源”中的“移动电源”是品类，不是品牌；Query “适用于海信
电视的回音壁”要求兼容海信电视，不要求回音壁品牌为海信。

hard/soft 规则：
- 明确品类、明确预算上限、否定要求、指定规格或选项属于 hard；
- “优先、最好、倾向、左右”等偏好属于 soft；
- candidates 中的 hardness_hint 只是代码初筛提示；最终 hard/soft 必须服从 Query
  原文措辞，“最好”等明确偏好不能被强制改成 hard；
- 无法可靠判断时使用 needs_review，不要强行二选一。

只输出一个 JSON 对象：
{
  "selected_constraints": [
    {
      "candidate_id": "c0001",
      "description": "非空、简短、人类可读且不扩写的新描述",
      "hardness": "hard | soft | needs_review",
      "query_quote": "非空且逐字存在于 Query 的连续原文",
      "selection_reason": "非空；说明该候选为何由这段原文直接支持"
    }
  ]
}
不要输出 Markdown、解释性前后缀或任何额外字段。
```

V4 Flash 实际收到的 User 消息只有：

```json
{
  "task_id": 8187,
  "query": "求一对卡通-永结同心款的高档酒红色木梳……价格在20元左右。",
  "candidates": ["代码生成的完整候选数组"]
}
```

### 3.3 代码再次收口

Flash 返回后，代码会检查：

- 只能引用已存在且不重复的 `candidate_id`；
- description、selection reason 非空；
- hardness 只能是 `hard / soft / needs_review`；
- Query quote 必须能回指 Query 的真实连续原文；
- task ID、task-data hash、query hash、extractor/model/prompt 版本完全一致；
- 底层 `field_path / operator / expected_value` 重新从代码候选复制，不能采用模型改写值。

如果 JSON 已解析但 Schema 不合法，系统最多进行有限次数的“只修 Schema”重试，
不会放宽约束或生成默认 Rubric。最终 Rubric 按 `task_id` 缓存，Baseline、SFT、
GRPO 共用同一份。

示例 task 8187 最终从 7 条候选中选出 5 条：

| Rubric | Hardness | 要求 |
|---|---|---|
| `r0001` | hard | 商品品类应为新娘配件 |
| `r0002` | hard | 商品应支持高档 |
| `r0003` | hard | 商品应支持卡通-永结同心 |
| `r0004` | hard | 应选择“【卡通-永结同心】2个装” |
| `r0005` | soft | 价格倾向于 20 元左右 |

正式评测为每个 Final-200 Clean 任务生成一个 Rubric bundle；该 bundle 在模型之间冻结共享。

## 4. Actor Rollout

每个 Actor 使用完全相同的 Collector、System Prompt、工具 Schema 和推理参数：

| 设置 | 值 |
|---|---|
| Environment / Reward | v2.1 / Reward v3 |
| 每题 Rollout | 1 |
| Temperature / top-p | `0.0 / 1.0` |
| 最大环境步数 | 35 |
| 每回合最大生成 token | 512 |
| Context / safety margin | `24,576 / 512` |
| Context compaction | 关闭 |
| Search observation | top 20，预算 1,536 token |
| Product detail observation | 预算 4,096 token |
| Generic fallback | 预算 768 token |

一次 Rollout 会保存用户 Query、Assistant 文本、工具调用、Actor 实际看到的投影后
Observation、Guard 拒绝、每步状态、终局结果和基础审计信息。

示例 task 8187 的 SFT Actor 共执行 10 步：搜索一次、打开一个候选、查看
Description/Features/Reviews、选择“【卡通-永结同心】2个装”、确认 variant
价格为 9 元并购买。

## 5. 代码硬检查

LLM Judge 之前先进行确定性预处理：

1. 将不同来源的原始 Rollout 规范化成固定 Schema；
2. 为动作尝试、已执行步骤和事件生成稳定 ID，例如 `a0001 / s0000 / e0001`；
3. 计算 Environment Reward 与终局事实；
4. 计算工具次数、步数、候选打开数和购买/放弃次数；
5. 检查 malformed tool call、Action Guard 拒绝、step error 和非法动作；
6. 统计重复动作、重复搜索和 environment repeat loop；
7. 统计 Observation 投影、截断、上下文 token 和 overflow；
8. 检查 release error、任务缺失及其他 `infrastructure_invalid` 情况。

如果轨迹被判为 `infrastructure_invalid`，系统直接生成 `not_judged`，不会要求 Pro
猜一个分数，但该任务仍留在固定 200 题分母中。

## 6. 第二位 LLM：V4 Pro 评价完整轨迹

### 6.1 Pro 实际能看到什么？

`deepseek-v4-pro` 的输入由以下内容组成：

- `task_id` 与 `trajectory_id`；
- 用户原始 Query；
- V4 Flash 已冻结的 Rubric；
- 五个维度及允许分值 `[0, 1, 2]`；
- 冻结的错误类型集合；
- Actor-visible trajectory：
  - Assistant 当时输出的文本；
  - 工具名、参数和环境 action；
  - Actor 当时真正看到的投影后 Observation；
  - Guard 拒绝与 step error；
  - 稳定的 event IDs；
- 仅包含 `done / over` 的中性终局状态；
- 白名单化的代码指标：
  - actions and efficiency；
  - repetition；
  - legality；
  - context；
- 要求输出的严格 JSON Schema。

Pro 明确看不到：

- raw Observation；
- Gold 商品私有字段和 Actor 未看到的候选；
- Reward v3 分数、reward type、hard gates、weighted score；
- strict success、purchase success 或代码给出的成功结论；
- infrastructure validity；
- 其他模型在同一题上的结果。

这种隔离防止 Pro 因为先看到 Reward 或 Gold 答案而倒推“轨迹一定正确”。

### 6.2 Pro 的完整 System Prompt

当前冻结版本为 `trajectory-judge-v1-draft-r3`：

```text
你是当前 Shopping Agent / ShopSimulator 项目的离线轨迹 Judge。

你必须只依据输入中的 actor_visible_trajectory 评价 Actor 的行为。不得假设你能看到
audit raw_observation、Gold 商品私有字段或未展示给 Actor 的候选。输入不会包含
Environment Reward、Reward 分项或代码判定的任务成功结论；这些结果由独立面板
负责，不能由你推断、覆盖或改写。

逐条需求状态只能是 satisfied、violated、unknown、not_applicable。没有可见证据时
使用 unknown。每项判断尽量引用真实 event_id；不得伪造不存在的 event_id。

五个维度分别打 0、1、2 分，不加权、不计算总分：
- search_strategy：搜索是否覆盖品类和关键条件，改写是否有效，是否机械重复；
- candidate_utilization：是否利用可见的高匹配候选，比较是否必要且不过度；
- evidence_verification：购买前是否核验关键属性、规格和最终价格；
- decision_quality：最终选择、规格和购买/放弃决策是否合理；
- termination_efficiency：是否过早购买/放弃、无效探索或耗尽步骤。

错误类型必须从输入提供的 frozen_error_taxonomy 中选择；没有主要错误时 primary
使用 null。只输出 JSON，不输出 Markdown。
schema_version 必须是 shopping-trajectory-judge-v1。禁止输出 total_score、overall_score
或任何综合分。
```

### 6.3 五个评分维度

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| Search Strategy | 核心条件缺失、无效或机械重复 | 初始搜索合理但改写/收敛一般 | 覆盖关键条件并能有效改写或翻页 |
| Candidate Utilization | 忽略明显高匹配候选 | 候选合理但比较不足或略冗余 | 识别高匹配候选并在必要比较后收敛 |
| Evidence Verification | 未核验关键属性/规格/价格 | 只核验部分关键要求 | 可靠核验所有决策关键证据 |
| Decision Quality | 违反硬约束或错误购买/放弃 | 基本合理但有未满足项或证据缺口 | 商品、规格与决策均有证据支持 |
| Termination Efficiency | 过早终止、循环或耗尽步骤 | 存在轻度冗余 | 证据充分后及时购买或合理放弃 |

Pro 还要逐条输出每个 Rubric 的
`satisfied / violated / unknown / not_applicable`、理由和证据 event IDs，并从冻结
taxonomy 中给出 primary/secondary errors。代码随后验证 Rubric IDs 和 event IDs
必须真实存在，五个维度必须齐全且只能为 0/1/2，禁止 Pro 输出总分。

示例 task 8187 的 SFT 轨迹得到：

| 维度 | 分数 | 主要理由 |
|---|---:|---|
| Search Strategy | 2 | 搜索覆盖卡通、永结同心、木梳、礼盒和酒红色 |
| Candidate Utilization | 1 | 所选商品合理，但只打开一个候选，比较不足 |
| Evidence Verification | 1 | 规格和价格已确认，但详情页证据有限 |
| Decision Quality | 2 | 硬约束、选项和预算均满足 |
| Termination Efficiency | 2 | 10 步内完成，无无效循环 |

五条 Rubric 均被判为 `satisfied`，没有 primary error。每一项结论都引用轨迹中的
`e0001`–`e0010` 事件，而不是引用隐藏 Gold。

## 7. 最后统计什么？

每道题最终拼装四个互不覆盖的面板：

### A. Reward 与终局

- Reward version/type/valid；
- strict gold success 与 purchase success；
- final reward、terminal utility、weighted score；
- done、over、termination reason；
- hard gates 和 Reward dimension scores。

### B. Query Rubric

- 每条需求的 satisfied/violated/unknown/not_applicable；
- 按 hard/soft 分组统计；
- Reward 与 Rubric 是否发生 disagreement。

Reward 与 Rubric 冲突时两者都保留。例如 Reward 判为 gold，但 Rubric 发现预算自然
语言约束违反时，记录 `reward_rubric_disagreement=true`，不让任一面板覆盖另一方。

### C. 轨迹质量

- Pro Judge 有效覆盖率；
- 五个维度各自的 0/1/2 分布与均值；
- primary/secondary error taxonomy 分布；
- 每条判断对应的 event IDs 和整体诊断。

### D. 确定性行为与基础设施

- executed steps 与 action attempts；
- 各工具调用次数；
- Guard 拒绝和非法动作；
- 重复动作与重复搜索；
- Observation 截断和上下文使用；
- infrastructure-invalid 数量及 task IDs。

汇总时始终以 183 为固定分母。最后按 `task_id` 对 Baseline、SFT、GRPO 做配对比较，
统计成功状态迁移、Reward type 迁移、hard violation 差值、五维分数差值、步数、
Guard 和重复动作变化；仍然不生成一个综合总分。

正式运行的 Pro Judge 覆盖率为：

| Actor | Valid Judge | Not judged | Coverage |
|---|---:|---:|---:|
| Baseline | 198 | 2 | 99.0% |
| SFT | 195 | 5 | 97.5% |
| GRPO step 100 | 195 | 5 | 97.5% |

## 8. 代码与产物

当前实现按职责拆分在：

```text
src/shopping_grpo/evaluation/
  trajectory.py       原始 Rollout 规范化和 event IDs
  metrics.py          确定性硬检查与行为指标
  task_facts.py       从 ShopSimulator 恢复私有任务事实
  rubric.py           候选提取与 Flash 输出物化
  prompts.py          两个冻结 Prompt 和 Judge-safe 输入
  model_client.py     OpenAI-compatible JSON 请求
  contracts.py        Rubric/Judge Schema 严格校验
  blind_guard.py      Final-200 Clean 内容与 task-ID 防泄漏
  results.py          四面板拼装和固定分母汇总
  comparison.py       Baseline/SFT/GRPO 配对比较
```

正式运行会产生：

```text
shared/task_facts.jsonl
shared/rubric_candidates.jsonl
shared/rubrics.jsonl
MODEL/trajectories.jsonl
MODEL/preprocessed.jsonl
MODEL/judge_requests.jsonl
MODEL/judges.jsonl
MODEL/evaluations.jsonl
MODEL/evaluation_summary.json
model_comparison.json
```

完整轨迹和 Judge 请求可能体积较大，因此属于 `outputs/` 运行产物；Git 中只提交
紧凑的配置与结果摘要。
