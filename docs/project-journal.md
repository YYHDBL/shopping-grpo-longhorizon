# 从能逛商店到能买对东西：一个 Shopping Agent 的实验札记

> 持续更新中。本文记录已经做过、已经验证过的工作，也记录踩过的坑；没有完成的 SFT 和 GRPO 会明确标为计划，而不是结果。
>
> 最近更新：2026-07-20｜当前阶段：数据冷启动完成，已获得第一份 Instruct 零样本基线

## 摘要

我想验证一件并不轻松、但足够具体的事：一个小模型能否先通过 SFT 学会在购物环境里稳定地搜索、核验和购买，再借助环境奖励继续优化？这不是一个“让模型推荐商品”的问答项目，而是一条可审计的 Agent 后训练链路。当前已经完成环境闭环、轨迹采集、确定性验收和训练数据构造；真正的 SFT、GRPO 和最终对比结果还没有发生。

实验过程不靠事后写漂亮结论。每一个阶段都会留下独立记录：任务、假设、配置、原始产物、失败样例、结论和下一步。总索引见 [实验记录目录](experiments/README.md)。

## 我想做的不是“回答购物问题”，而是让模型真正买对东西

这个项目从一个很具体的目标出发：让语言模型在文字版购物环境里，接到一次完整需求后，自己搜索商品、打开详情页、选择规格，最后完成购买。这里的“单轮”是指用户只在开始说一次需求；模型在商店里仍然要进行很多步行动。

我参考的是 [ShopSimulator 论文](https://arxiv.org/abs/2601.18225)。它把购物 Agent 看成一个长轨迹决策问题：搜索结果里有相似商品，商品页上有属性、规格和动态价格，买错一个选项也算失败。论文的核心启发并不是某个神奇算法，而是顺序：**先让 Agent 学会稳定使用工具，再用环境奖励继续优化。**

因此本项目先把问题收窄：暂不做用户多轮澄清、Persona、Reward Model、LLM Judge 或复杂 Agent 框架，只做单轮、多步、可购买的 ShopSimulator 任务。这样可以先把环境、轨迹、数据质量和评测闭环打通；未来要扩展到多轮时，工具协议和轨迹结构仍可复用。

```text
ShopSimulator 单轮任务
        ↓
强 Teacher 在真实环境里 rollout
        ↓
规则验收与清洗
        ↓
OpenAI tool-calling SFT 数据
        ↓
LoRA SFT（计划中）
        ↓
veRL Vanilla GRPO（计划中）
```

## 环境和任务从哪里来

购物世界不是我手工编题或拼商品描述。项目使用 [ShopAgent-Team/ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator) 提供的本地环境。论文中的完整环境覆盖 12 个中文电商领域和百万级商品；当前本地安装的轻量任务集包含 **23,421** 个可用任务。每次 `reset(task_id)` 后，环境才把该任务的用户需求和首个 observation 返回给 Agent。

这点很重要：训练仓库只保存 task id，而不把隐藏的 goal、标准答案、reward 明细提前写进 prompt。它们只存在于环境的终局反馈中，用于验收。这样模型学习的是“看到用户需求和页面以后该怎样行动”，而不是背答案。

ShopSimulator 被独立放在自己的 Python 3.10 环境中；训练、推理和数据构造放在本仓库的 Python 3.12 环境中。两个仓库通过结构化 HTTP 接口 `/api/shop_agent` 通信，而不是解析网页 HTML。每条 trajectory 独占一个环境租约，完成、报错或中断时都尝试 `release_one`。这让“模型输出坏了”和“环境服务坏了”能被分开记录。

## 一条轨迹是怎样被采出来的

Teacher rollout 不使用 LangChain、AutoGen 一类的 Agent 框架。它就是一个很直白的循环：模型根据 messages 生成标准 OpenAI function call，脚本把调用转为 ShopSimulator action，拿到 observation 后再交还给模型。现在的统一工具集有 12 个工具，包括搜索、打开商品、选择规格、查看详情/属性/评论、翻页、返回搜索和购买。SFT、基线评测与未来 GRPO 将使用同一份 schema。

为了让轨迹能被调试，原始记录会保留 messages、每一步工具调用、实际环境 action、observation、终局 reward 和异常信息；为了让训练数据不泄漏答案，构造 SFT JSONL 时会移除 goal、reward_detail、终局隐藏反馈及运行时守卫信息。最终每行是标准的 `messages + tools`，可交给 Hugging Face 的 chat template 渲染。

这条路并不总是“模型多跑几次就会变好”。我们在小批量里反复见到三类问题：无参数工具带垃圾字段、选完规格还返回搜索换商品、以及为了赶步数而直接买下不满足约束的商品。相应地，采集器只允许每回合执行一个工具调用；工具参数必须符合 schema；点击和规格值必须来自最新页面；一旦开始选规格，轨迹不能再回到搜索阶段。提示词也明确要求先核验硬约束，再购买。

## 当前数据快照：不是“全都收下”，而是只保留能证明买对的轨迹

截至本次更新，公开快照位于 `outputs/flash_accepted_500_parallel/`：

| 项目 | 数量 | 说明 |
|---|---:|---|
| 原始轨迹 | 757 | 成功、失败和异常都保留，供排查和断点续跑 |
| accepted | 380 | 通过确定性验收的购买轨迹 |
| rejected | 377 | 保留结构化拒绝原因，不假装它们是训练样本 |
| 可训练 SFT 行 | 380 | OpenAI messages + tools 格式 |
| 24K chat-template 预检可用 | 353 | 其余 27 条过长，训练时会丢弃而非截断半段工具调用 |

“accepted”不等于模型说自己完成了。每条轨迹必须：没有未处理异常、工具调用格式合法、实际执行过购买、环境确认 `done` 与 `over`，并且类别、属性、规格、价格四项 reward（`r_type`、`r_att`、`r_option`、`r_price`）都等于 1。这个标准比较苛刻，但它避免把“买到了一个东西”误当成“完成了用户任务”。

377 条拒绝轨迹也很有价值：最常见的是没能结束环境或没有购买；另一些是属性、规格或价格不符合任务。它提醒我，后续提高 accepted 数量的重点不只是扩大调用次数，更是提高 Agent 的搜索、核验和选择能力。

## 从几次卡死里学到的工程课

最早的服务问题并不在模型能力，而在边界条件。ShopSimulator 的单环境租约能够正常 `reset → release_one → reset`；真正导致评测卡住的是模型客户端没有给单次回答设置 `max_tokens`。`max-model-len` 限制的是总上下文，不会阻止模型在没调用工具时持续输出纯文本。现在每回合固定 `max_tokens=512`，同时保持 `max_steps=35`。

另一个容易混淆的点是 Qwen3.5 的命名。`Qwen/Qwen3.5-2B` 是带 chat template、可调用工具的 Instruct 模型；裸预训练权重才是带 `-Base` 后缀的版本。服务器端的最小验证已经确认：传入本项目 12 个工具 schema，并使用 vLLM 的 `qwen3_coder` parser 后，模型能在 26 个生成 token 内返回 `search_products` 工具调用。错误的 Hermes parser 曾导致 HTTP 400 和纯文本行为，这类配置问题不能归咎于模型本身。

## 基线怎么测：固定任务，而不是“今天挑 200 条、明天挑另 200 条”

我从 23,421 个 task id 中排除了当前 SFT 用过的 380 个任务，再以固定随机种子 `20260720` 抽取了 200 条，形成不可改写的 `data/benchmarks/shop_benchmark_v1.jsonl`。它不是训练集，也不能被拿去继续采集 SFT 或 GRPO 轨迹。

Base（这里指“尚未做购物 SFT 的 Instruct 模型”）、未来 SFT adapter 和未来 GRPO checkpoint 都必须使用同一份任务、同一 system prompt、同一工具 schema、`temperature=0`、`max_steps=35`、`max_tokens=512`，每题只 rollout 一次。主指标是严格成功率，以全部 200 题为分母；没有跑完的题也算失败。辅助指标包括环境完成率、四个 reward 分量的通过率、平均工具步数和动作守卫拒绝原因。

第一份完整基线已经跑完（commit `c04caf1`）：Qwen3.5-2B Instruct 完成了全部 200 个 task，但严格成功率为 **0%**，仅 1 条轨迹让环境结束——且该轨迹买错了品类。

深度分析揭示了三种明确的失败模式，按严重程度排列：

1. **打开商品后立刻搜索（57 条）**：模型不理解"商品详情页不支持搜索"——这是最典型的不会读 observation 的症状
2. **盲点不存在的 Features/Attributes 按钮（51 条）**：模型猜测每个商品都有这些子页，而不是先确认按钮存在
3. **无限翻页不打开商品（10 条）**：模型在搜索结果页一直翻页直到步数耗尽
4. **HTTP 400 错误（27 条）**：这些轨迹在报错前平均执行了 15.7 步正确 tool call，提示上下文增长可能导致 vLLM 服务端错误，不应与模型策略能力混淆

92.2% 的失败轨迹在**前 3 步**就被首次拦截，average_steps 仅 6.7。这不是"长链推理难"，而是最基本的"读当前页面 → 判断可用操作 → 选择正确按钮"环节全线崩溃。

核心发现：Qwen3.5-2B Instruct 会生成合法 tool call JSON，但**完全不懂购物场景下的工具使用规则**。它不知道搜索结果页和商品详情页有哪些不同的按钮，也不知道 `view_features` 不是每个商品都有的。这是 SFT 冷启动必要性的直接、可量化的证据。

完整分析和可检验假设见 [实验 01](experiments/01-qwen35-2b-instruct-baseline-2026-07-20.md)。

## 接下来，但还没有发生的事

下一步是用 Qwen3.5-2B 做 LoRA SFT。已有训练脚本会通过目标模型的 chat template 渲染样本，只计算 assistant 文本与 tool call 的 loss，用户需求和环境 observation 不参与 loss；具体 batch size、学习率、epoch 和 adapter 配置会在 GPU smoke test 后再决定，而不会先拍脑袋写进结论。

再往后才是 GRPO。这里会借鉴 [agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon) 的实验纪律：独立评测、避免训练任务泄漏、先跑 Vanilla GRPO 基线再做消融。但我们不会在第一版就照搬它针对 τ-bench 的 PRM-Lite、LATA 等额外机制。本项目的第一目标是用 ShopSimulator 的环境终局严格 reward 跑通 **Vanilla GRPO**，确认 SFT 是否提供了足够的冷启动策略；只有看到真实的奖励饱和、长轨迹退化或训练崩溃，才决定是否有必要增加复杂度。

---

### 更新约定

这篇文章负责讲清楚项目为什么这样设计、一路遇到了什么问题；细粒度的命令、配置、结果和失败样例写入 `docs/experiments/`。后续每完成一个可复现实验，就追加对应记录：运行配置、数据/模型版本、固定 benchmark 结果、失败样例和下一步判断。这样这份文档既是项目对外的故事，也是一份不给未来自己挖坑的实验日志。
