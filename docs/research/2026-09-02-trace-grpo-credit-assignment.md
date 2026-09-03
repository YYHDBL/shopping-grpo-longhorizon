# TRACE 用于 Shopping GRPO 的可行性研究

## 结论

可以做实验，但不建议现在直接把现有 GRPO 全量替换成 TRACE。更稳妥的定位是：**保留现有终局 Reward v3 和 GRPO 更新，只增加一个 turn-level 辅助 advantage，先做小规模严格配对实验**。

结构上，这个项目是合适的：它有最长 35 步的多轮工具轨迹、稀疏终局奖励和明确的 assistant/tool 边界，正是 TRACE 要解决的长程 credit assignment 场景（见本项目的 [GRPO 文档](../grpo.md) 与 [Reward v3](../reward-v3.md)）。但语义上不是即插即用：TRACE 的论文实验依赖**短、唯一、可验证的 gold answer**，购物任务的目标却是结构化购买决策，而且 Reward v3 接受满足约束的替代商品。论文也明确把长结构化、多解或开放目标列为当前方法的适用边界。[论文 §6](https://arxiv.org/pdf/2607.13988v1#page=12)

因此，是否值得继续实现取决于一个先验诊断：冻结参考模型对“gold 购买目标”的 prefix log-prob，是否真的随有效购物进度上升，并与 Reward v3 终局结果正相关。若这个 proxy 不成立，照搬 TRACE 公式只会给 GRPO 加噪声。

## 名称与来源核实

- 正确名称是 **TRACE**，全称 **Turn-level Reward Assignment via Credit Estimation**，不是 Trave。[arXiv 摘要页](https://arxiv.org/abs/2607.13988v1)
- 论文由 UW–Madison 与 Microsoft Research 作者提出；v1 于 2026-07-15 提交。[论文首页](https://arxiv.org/pdf/2607.13988v1#page=1)
- 用户给出的 [`alphaXiv/trace-turn-level-reward-assignment-via-credit-es`](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es) **不是作者官方代码**。其 README 明确自称 “bounded, public reproduction”，并给出 “partially reproduced” 的结论。[第三方仓库 README](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es#trace-turn-level-reward-reproduction)
- 论文正文和 arXiv 元数据没有提供作者 GitHub/code URL。因此当前能确认的一手算法来源是论文；alphaXiv 仓库只能作为第三方最小实现参考，不能当作论文原实现。[arXiv 条目](https://arxiv.org/abs/2607.13988v1)

## TRACE 相对 GRPO 改了什么

TRACE 没有替换 GRPO optimizer，也没有训练 critic。它保留 GRPO 的终局 group-relative advantage，再给每个工具 turn 增加基于 frozen reference model 的局部 credit。[论文 Algorithm 1 与 §3](https://arxiv.org/pdf/2607.13988v1#page=4)

对同一 prompt 的第 \(g\) 条 rollout，先照常用终局奖励构造：

\[
A^{out}_g =
\begin{cases}
(R_g-\bar R)/\sigma_R, & \sigma_R>0\\
0, & \sigma_R=0
\end{cases}
\]

然后在每个工具调用后的 observation 边界切出 prefix state \(S_k\)。用初始化 policy 的冻结副本 \(\pi_{ref}\) teacher-force gold answer \(y^*\)，计算平均 gold-token log-prob：

\[
\bar\ell_k=\frac{1}{|y^*|}\sum_t \log \pi_{ref}(y_t^*\mid S_k,y_{<t}^*)
\]

把它转成 remaining gap 和状态值：

\[
d_k=-\bar\ell_k+\epsilon,\qquad
V(S_k)=\log\frac{d_0}{d_k}
\]

相邻工具 turn 的一步 credit 是：

\[
\delta_k=V(S_{k+1})-V(S_k)=\log\frac{d_k}{d_{k+1}}
\]

正值表示该工具动作和 observation 让 gold answer 更容易预测，负值表示它把轨迹带离目标。一步 credit 会 telescope，所以单纯插入冗余 turn 不能增加累计 credit。[论文 Eq. 5–7](https://arxiv.org/pdf/2607.13988v1#page=4)

为处理 search 后要到后续 open/find 才出现证据的延迟效果，论文使用归一化的 \(K\)-step backup：

\[
c^{(K)}_{g,k}=
\frac{\sum_{u=k}^{h_{g,k}}\gamma_{td}^{u-k}\delta_{g,u}}
{\sum_{u=k}^{h_{g,k}}\gamma_{td}^{u-k}},
\quad h_{g,k}=\min(k+K-1,T_g-1)
\]

如果这个窗口触及轨迹末尾，再加入终局锚点：

\[
r^{turn}_{g,k}=c^{(K)}_{g,k}
+\mathbf{1}[h_{g,k}=T_g-1]\lambda_{term}\gamma_{td}^{T_g-k}A^{out}_g
\]

最终，属于工具 turn \(k\) 的 assistant token 使用混合 advantage：

\[
\hat A_{g,t}=\alpha_{out}A^{out}_g+\alpha_{turn}r^{turn}_{g,k}
\]

tool observation token 继续 mask；turn credit 不做 group normalization。之后仍然进入原来的 token-level clipped GRPO objective。[论文 Eq. 8–12](https://arxiv.org/pdf/2607.13988v1#page=6)

换句话说，最小移植不是新建一套 trainer，而是给现有 GRPO 增加：**prefix scorer → turn credit → token/turn advantage 映射**。

## 训练数据和 rollout 的硬要求

TRACE 训练至少需要以下信息：[论文 Algorithm 1](https://arxiv.org/pdf/2607.13988v1#page=5)

1. 每个训练 prompt 有训练时可见的 gold target \(y^*\)。论文使用短答案 normalized exact match，并额外给合法 answer format 0.1 分。[论文实验设置](https://arxiv.org/pdf/2607.13988v1#page=7)
2. 每条 rollout 保留完整 action/observation transcript，并能形成 \(S_0,\ldots,S_T\) 的工具边界 prefix。
3. assistant token 能映射回产生它的工具 turn；tool observation token 不参与 policy-gradient loss。
4. 同一 prompt 仍需多个 rollout 来计算 GRPO 的终局 group-relative advantage。论文使用 \(G=8\)；TRACE 的 dense credit 则是单轨迹内部信号，不做跨 rollout 标准化。[训练超参表](https://arxiv.org/pdf/2607.13988v1#page=19)
5. 一个冻结 reference checkpoint。论文发现初始化 checkpoint 已足够，换成 step-200 checkpoint 只从 35.6 变为 36.1，不需要额外强 teacher。[参考模型消融](https://arxiv.org/pdf/2607.13988v1#page=10)

现有 Shopping GRPO 已有多轮 tool transcript、终局 Reward v3、每 prompt 4 个 rollout，并启用了 raw chat 返回，因此轨迹侧的大部分前提已经具备。[现有 GRPO 配置](../../configs/grpo.yaml) 真正缺少的是：一个与 Reward v3 目标一致、可由 reference model 稳定评分的 canonical gold target，以及训练期的 prefix log-prob scorer。

## 关键超参

论文报告的 TRACE 默认值如下：[论文 §4.1](https://arxiv.org/pdf/2607.13988v1#page=7)、[附录训练表](https://arxiv.org/pdf/2607.13988v1#page=20)

| 参数 | 论文值 | 作用 |
|---|---:|---|
| \(\alpha_{out}\) | 1.0 | 保留终局 GRPO 信号 |
| \(\alpha_{turn}\) | 0.2 | turn credit 辅助权重 |
| \(\epsilon_{train}\) | 0.1 | remaining gap 数值稳定项 |
| \(K\) | 3 | TD look-ahead horizon |
| \(\gamma_{td}\) | 0.8 | 延迟 credit 折扣 |
| \(\lambda_{term}\) | 2.0 | terminal fill 强度 |
| turn advantage normalization | none | 不做组内/组后标准化 |
| advantage clipping | 0.0 | 不额外裁剪 turn advantage |

配套 GRPO 设置是：8 rollouts/prompt、global batch 128、Adam constant LR \(10^{-6}\)、weight decay 0.01、clip lower/upper 0.20/0.28、KL=0、entropy=0、per-token loss。rollout 最长 60 tool turns、48k trajectory tokens。[完整超参表](https://arxiv.org/pdf/2607.13988v1#page=19)

论文消融表明 \(\alpha_{turn}\) 和 \(K\) 都不能盲目加大：过大的局部 credit 会压过终局正确性，过长的 look-ahead 会把不相关后续噪声传播回来。[论文 §4.4](https://arxiv.org/pdf/2607.13988v1#page=10)

## 训练和推理开销

- 不需要额外训练 critic、process reward model、step labels、强 LLM judge 或 Monte Carlo continuation；reference model 只前向、不更新。[论文摘要与 §5](https://arxiv.org/pdf/2607.13988v1#page=1)
- 仍有实质新增训练开销：一条 \(T\)-turn rollout 要对 \(T+1\) 个 prefix 评分，每个 prefix 都 teacher-force gold answer token。论文称同一轨迹的 prefix 分数可在一次 batched forward 中得到；30B 的脚本设置需要远程 reference-model scoring endpoint。[论文 §3.2](https://arxiv.org/pdf/2607.13988v1#page=4)、[附录 A.1](https://arxiv.org/pdf/2607.13988v1#page=19)
- 论文没有报告 TRACE 相对 GRPO 的 wall-clock、GPU-hour 或吞吐差值，不能从“没有 critic”推断训练成本很低。
- 推理阶段不需要 prefix scorer 或 turn reward；因此根据论文训练流程可推断，导出的 policy 与 GRPO 一样独立推理，没有额外部署路径。这个结论是对算法流程的推断，不是论文给出的 wall-clock 实测。[论文 Algorithm 1](https://arxiv.org/pdf/2607.13988v1#page=5)

第三方缩小复现的 model time 差异很不稳定：Instruct 配对是 1515s vs 1027s（TRACE 约 +47.5%），exact Thinking 配对是 18926s vs 18507s（约 +2.3%）。这只能说明开销高度依赖实现瓶颈，不能外推到论文或本项目。[第三方实验日志](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es#experiment-log)

## 论文实验结论

论文在相同 backbone、数据、browser interface、terminal reward 和评测协议下对比了 TRACE 与 outcome-only GRPO：[论文 Table 1](https://arxiv.org/pdf/2607.13988v1#page=8)

| Backbone | 方法 | BrowseComp-Plus | BrowseComp | GAIA | xbench-DeepSearch | 平均 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | GRPO | 30.0 | 5.1 | 38.8 | 44.0 | 29.5 |
| Qwen3-4B | TRACE | **35.6** | **6.7** | **44.6** | **49.0** | **34.0** |
| Qwen3-30B-A3B | GRPO | 36.4 | 10.8 | 45.6 | 37.0 | 32.5 |
| Qwen3-30B-A3B | TRACE | **42.6** | **12.9** | **52.0** | **45.0** | **38.1** |

作者还报告 TRACE 学习曲线更早上升、收敛更快，30B 的 step-160 TRACE checkpoint 已超过 step-200 outcome baseline。[论文学习曲线分析](https://arxiv.org/pdf/2607.13988v1#page=9)

credit 形式消融中，BrowseComp-Plus 从 GRPO 30.0 提升到 raw delta 32.4、remaining-gap 34.6、log-ratio 35.5。作者另对 830 条 rollout、3742 个 tool turns 做离线诊断，log-ratio 与最终 reference score、正终局结果的相关系数分别为 0.751、0.713，pairwise ranking accuracy 为 98.24%。[论文 §4.4](https://arxiv.org/pdf/2607.13988v1#page=10)、[附录 A.2](https://arxiv.org/pdf/2607.13988v1#page=20)

这些结果只直接支持“短答案长程搜索”。论文没有在 shopping agent 上实测；它只在背景中把 shopping/navigation 列为 agent 应用。因此不能把上表的收益幅度当作本项目的预期提升。[论文实验范围与限制](https://arxiv.org/pdf/2607.13988v1#page=12)

## 用户给出的第三方代码能借鉴什么

第三方仓库把完整缩小实验放在一个 [`trace_repro.py`](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/851fb37a60a6e421b67d367fb43aa91f719127a9/trace_repro.py) 中：

- rollout 保存 prefix boundaries 和 `turn_index`；
- `answer_score()` 用禁用 LoRA adapter 的 base model 计算 gold-answer log-prob；
- `trace_credits()` 计算 log-ratio delta 和归一化 \(K\)-step credit；
- `group_advantages()` 计算 outcome advantage；
- 训练循环把 outcome 与 turn advantage 混合。

核心参考位置是 [`answer_score()` / `trace_credits()`](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/851fb37a60a6e421b67d367fb43aa91f719127a9/trace_repro.py#L291-L326) 和 [advantage 混合](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/851fb37a60a6e421b67d367fb43aa91f719127a9/trace_repro.py#L360-L430)，参数在 [`config.json`](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/851fb37a60a6e421b67d367fb43aa91f719127a9/config.json)。

但它不是可直接移植的论文实现：只跑 4 turns、4 rollouts、12 updates、16 train/8 held-out questions，使用 query-local 小语料、LoRA 和 group-relative REINFORCE，而不是论文的 60 turns、8 rollouts、200 updates、完整 index 和 clipped GRPO；它还逐 prefix 调用 scorer，没有实现论文所说的单次批处理。[第三方详细报告](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/851fb37a60a6e421b67d367fb43aa91f719127a9/reports/trace-reproduction/report.md)

它的 exact-checkpoint 配对结果是 TRACE 6.25% vs outcome-only 9.38%，没有复现论文的成功率收益；Instruct 配对也没有成功率收益。由于实验规模和 optimizer 都不同，这既不能证伪论文，也提醒我们必须先做本项目自己的严格对照。[第三方 README 结论](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es#trace-turn-level-reward-reproduction)

## 对 Shopping GRPO 的具体判断

### 合适的部分

1. 现有任务最长 35 步，搜索、打开商品、检查详情、选择规格到最终购买之间存在明显延迟 credit；这比短推理更符合 TRACE 的动机。[Reward v3 终止规则](../reward-v3.md)
2. 现有 GRPO 只把终局 Reward v3 传播给整条 rollout。一个失败轨迹仍可能包含正确搜索和候选排查，TRACE 有机会避免把这些有效 turn 与最后一次错误购买一起惩罚。这正是论文定义的问题。[论文引言](https://arxiv.org/pdf/2607.13988v1#page=2)
3. 项目已有 4 rollouts/prompt、完整生成诊断、工具序列和 reward breakdown，适合做 matched A/B 与离线 prefix-credit 审计。[现有 GRPO 文档](../grpo.md)
4. 现有 frozen reference 路径和 policy checkpoint 概念与 TRACE 接近；无需再引入一个可训练 critic。

### 不直接合适的部分

1. Reward v3 的成功不是一个短文本答案，而是环境中的结构化购买终局；模型通常通过 `buy_now` tool call 完成，而不是输出论文式 `<answer>`。
2. Reward v3 同时接受 exact `gold_purchase` 和 `valid_alternative_purchase`。若把唯一 gold ASIN/option 串作为 \(y^*\)，reference scorer 会天然偏向 exact target，即使另一商品完全满足约束并获得正终局奖励。这会让 turn credit 与现有 verifier 冲突。[Reward v3 的成功定义](../reward-v3.md)
3. 轨迹中的 observation 很长，现有最大模型长度 24,576；对每个 tool boundary 重复构造 prefix 会显著增加 scorer token-volume 和显存压力。[现有 GRPO 配置](../../configs/grpo.yaml)
4. 现有 rollout group size 是 4，不是论文的 8。可以先保持 4 做严格 matched 实验，但不能预期复现论文绝对数值。

## 最小实验方案

### Phase 0：只做离线 proxy 验证，不训练

从现有 `training_diagnostics.jsonl` 抽取一批包含成功、部分成功、失败的完整 rollout。在每个工具边界，用 frozen SFT 初始化模型评分一个固定 canonical target。先用最小、可审计的 target 表达：`buy_now` 的 gold ASIN 与必要 options 的规范化工具调用串。

至少检查：

- \(V(S_T)-V(S_0)\) 与 Reward v3 `terminal_utility`、strict success 的相关性；
- 成功轨迹中，关键的 search/open/select_option 后是否通常出现正 \(\delta_k\)；
- `valid_alternative_purchase` 上是否系统性出现负 credit；
- 空转、重复和错误候选转移是否通常是零或负 credit；
- prefix scorer 的 token 数、wall time 和峰值显存。

继续标准：相关方向稳定，且不会系统性惩罚 Reward v3 认可的替代购买。若最后一项冲突明显，就不要直接上 gold-answer TRACE；应先研究 structured-goal value estimator，这已经超出论文原方法。

### Phase 1：两臂小规模训练

只开两个严格匹配实验臂，复用相同初始化 checkpoint、训练 prompts、采样参数、rollout 数、训练步数和评测集：

- Control：当前 GRPO。
- Treatment：GRPO + TRACE turn advantage。

首轮保持现有 4 rollouts/prompt，不为模仿论文先扩大算力。TRACE 起始参数直接用论文值：\(\alpha_{out}=1.0\)、\(\alpha_{turn}=0.2\)、\(\epsilon=0.1\)、\(K=3\)、\(\gamma=0.8\)、\(\lambda_{term}=2.0\)。只跑足以观察学习曲线的短实验；若 Phase 0 信号不好，不进入这一阶段。

实现边界应限制在四处：

1. rollout 产物补充 tool-boundary prefix 与 token-to-turn map；
2. frozen reference 批量 gold-target log-prob scorer；
3. 一个纯函数计算 \(\delta\)、\(K\)-step credit、terminal fill 和 mixed advantage；
4. 在现有 veRL GRPO loss 前注入 per-token/per-turn advantage。

不新增 critic、不改 Reward v3、不改环境、不另建 optimizer。第三方单文件实现只能抄公式和数据流，不能复制它的 REINFORCE 训练循环。

### Phase 2：通过后再做一个必要消融

如果 Treatment 明显优于 Control，再增加一个 `raw delta` 或 \(\alpha_{turn}=0.1\) 消融，用来区分“任何 dense signal 都有效”与“TRACE log-ratio 设计有效”。第一轮不铺开超参网格。

### 评估指标

主指标仍应服从项目当前 contract：Final-200 Clean 的 strict success / purchase success / Reward v3 mean。额外记录：

- learning curve 达到同等 validation reward 所需 steps；
- turn-credit 均值、方差、正负比例和按 tool 类型分布；
- TRACE 与 outcome advantage 的尺度比；
- group 有效率和 skipped updates；
- 平均 tool turns、repeat/no-progress、early abstain；
- reference scoring 的训练 wall time、GPU memory、tokens/s。

最终判断不能只看 training reward。TRACE 论文声称的是更早学习和更好 held-out 表现；本项目也应以 matched validation / Final-200 结果为准。[论文学习动态](https://arxiv.org/pdf/2607.13988v1#page=9)

## 建议的决策

**Go：做 Phase 0 + 两臂小实验。** 这个任务的长工具链与稀疏终局奖励确实适合研究 turn-level credit。

**No-Go：暂不做全量 TRACE 替换。** 当前最大风险不是代码，而是 gold-answer log-prob 是否能代表结构化、多解的购物进度；论文没有验证这个前提，第三方复现也没有稳定复现收益。
