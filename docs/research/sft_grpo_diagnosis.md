# SFT 高增益、GRPO 低增益与逐 Turn 信用分配：诊断性调研

> 调研日期：2026-08-10。本文只做现有证据梳理和可证伪分析，不提出已经被结果证明的训练结论，也不执行训练。

## 结论先行

1. 用户口述的“Super Similar”几乎可以由仓库中的明确引用消歧为 **ShopSimulator**，不是另一个叫 Super Similar 的基准：项目 README、内嵌上游信息和论文链接都指向 `ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants`（arXiv:2601.18225）。仓库中没有检索到 “Super Similar” 这个名称。[项目 README](../../README.md)；[内嵌上游信息](../../environments/ShopSimulator/EMBEDDED_SOURCE.json)；[ShopSimulator 原论文](https://arxiv.org/abs/2601.18225)
2. 当前的 `60.5%` SFT 严格成功率**不能与论文“最强模型总体 full-success 低于 40%”直接比较**。论文总体分数平均了单轮、个性化单轮、多轮、个性化多轮四种场景；当前项目则明确规定为“完整需求在开头给出”的单轮任务，禁止追问，另有详细流程提示、强类型工具和 Action Guard。论文中 GPT-5 的单轮 `Rsucc` 本身已是 `40.78%`，总体才是 `32.65%`。[ShopSimulator §2.4、§3.1、Table 2](https://arxiv.org/pdf/2601.18225)；[当前 rollout prompt](../../src/shopping_grpo/evaluation/rollout.py)
3. 当前数据更支持“SFT 主要解决了协议与流程冷启动”，还不能支持“Reward 太简单”。Base→SFT 时 `done` 从 `18.0%` 到 `96.5%`、Guard rejection 从 `752` 降到 `52`，严格成功从 `0` 到 `121/200`。这首先说明 Base 几乎不会稳定执行协议；SFT 的大增益不等于细粒度商品判断已经容易。[实验对比](../../experiments/comparison.md)
4. `SFT 60.5% → GRPO 62.0%` 目前不是可靠的“仅提升 1.5 个点”结论：每题只有一次确定性 rollout；配对结果是 12 个失败转成功、9 个成功回退，净增 3 题。对 21 个 discordant pair 做双侧 exact McNemar 检验得到 `p≈0.664`，无法区分真实小增益与任务级抽样噪声。同时，平均 Reward、Guard rejection、循环和最大步数均有改善，说明“成功率没有明显拉开”和“训练完全没有作用”也不是同一件事。[实验对比](../../experiments/comparison.md)；[项目结果解释](../../docs/drafts/2026-07-30-shopping-agent-post-training-blog.md)
5. “SFT 训过头导致低熵、失去可塑性”是合理假设，但当前配置和结果**没有直接熵证据**：`calculate_entropy=false`，且没有提交逐 step 训练日志。应先看 rollout 多样性、零方差 group 比例、upper-clip fraction 和不同 SFT checkpoint 的 RL 可学习性，再决定减少 SFT epoch 或换“不完美 SFT”。[GRPO 配置](../../configs/grpo.yaml)
6. DAPO 并不是四个开关全都缺失：本项目已经用了 `token-mean` loss 和有限 dynamic sampling；真正未采用的是不对称 `clip-high`，而当前“长度 shaping”也不是 DAPO 的 token 截断处理。先补诊断指标，再做最小消融，比直接堆 DAPO 更有信息量。[DAPO 论文](https://arxiv.org/abs/2503.14476)；[DAPO 官方仓库](https://github.com/BytedTsinghua-SIA/DAPO)；[veRL 官方 DAPO recipe](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen2.5_32b_npu.sh)
7. AgentOPSD 确实做逐 Agent Turn 信用分配，但不是“把终局 Reward 平均拆到每 Turn”。它需要一个**训练期专用、推理期移除的 privileged skill**，用同一模型对同一已采样动作做普通/skill-conditioned 两次打分，再递归更新成功 belief。当前项目没有这样的 skill retriever；论文官方仓库截至调研日仍写着 “Code coming soon”，因此它适合作为第二阶段研究方向，不是现在可直接套用的低风险配置项。[AgentOPSD 论文](https://arxiv.org/pdf/2608.05987)；[官方仓库](https://github.com/ZethWang/AgentOPSD)

## 0. `data_new/` 是否可以替换当前数据

### 0.1 机械兼容：可以整套替换

只读审计确认：

- SFT 从 `379/49` 增加到 `800/200`，每条仍是当前的 `trajectory_id/task_id/messages/tools` schema；1,000 条轨迹的工具 schema 与仓库当前 `SHOP_TOOL_SCHEMAS` 完全一致，tool-call 参数均为合法 JSON object，assistant/tool call id 能配对，每条都有 `buy_now`。
- 新 SFT train/validation、新 GRPO train/validation、Final-200 之间的 `task_id` 两两零重叠；各自也无重复 task。
- `environment.json`、`evaluation/tasks.jsonl`、`evaluation/metadata.json` 与当前版本逐字节相同；Final-200 SHA-256 仍为 `2c4ff070...e0208`。
- 新 SFT JSONL 与新 GRPO Parquet 的实际 SHA-256 均匹配各自 metadata；Parquet footer 仍暴露相同的 `data_source/prompt/ability/reward_model/extra_info.task_id` Arrow schema。

但不能只替换 `sft/`：新 SFT 与旧 GRPO pool 有 **424 个 task_id 重叠**。所以若采用，必须把 `sft/` 与配套重建的 `grpo/` 一起原子替换。

### 0.2 实验可比性：暂不建议直接晋升为新基线

| 指标 | 当前 SFT | 新 SFT |
|---|---:|---:|
| 轨迹数 | 428 | 1,000 |
| 平均工具步数 | 9.72 | 7.50 |
| P90 工具步数 | 17 | 9 |
| 唯一工具路径 | 109 | 81 |
| 工具路径归一化熵 | 0.775 | 0.603 |
| 最常见单一路径占比 | 17.5% | 30.8% |
| assistant 空 content 占比 | 13.9% | 46.5% |
| 第一个打开商品位于首搜 top-3 | 79.1% | 83.0% |

新数据“更多、轨迹更短”，但动作路径反而更集中；它未必提供更宽的探索支持。metadata 中新旧教师名称也都写 `deepseek-v4-flash`，没有模型 revision，因此现有 provenance 不能证明“用了更好的教师版本”。新采集的 2,498 条 raw 中验收 1,026 条（41.1%），并记录 716 条 guard violation；旧文档记录的是 604 条中验收 428 条（70.9%）、guard violation 为 0。采样任务和采集条件不同，验收率不能直接当教师能力排名，但至少不能据此认定新教师更强。

更重要的是，旧 GRPO pool 来自 2,000 题难度 probe，并保留 short/medium/long 分层；新 pool 明确写着“未做难度探测”，JSONL 也只有 `task_id`。因此整套替换会同时改变 SFT 数量、SFT 行为分布和 GRPO 课程分布。若后续结果变化，三者无法归因。

晋升前的最小门槛是：补齐可复现构建脚本/真实 source path 与教师 revision；对新旧 SFT 做同配方、同开发集的 checkpoint 对比；决定是否恢复 GRPO 难度分层。当前 `data_new/README.md` 声称由 `scripts/build_data_new.py` 生成，但工作区内没有该脚本，metadata 所指的 `outputs/teacher_collection/raw_trajectories.jsonl` 也不存在。

## 1. 为什么当前 SFT 看起来远强于 ShopSimulator 论文中的模型

### 1.1 不是同一个任务协议

| 维度 | ShopSimulator 原论文 | 当前项目 |
|---|---|---|
| 场景 | 单轮、个性化单轮、多轮、个性化多轮 | 单轮；完整需求开头给出 |
| 用户交互 | 多轮场景要主动澄清，购买前还要向 Shopper 确认 | 明确禁止追问，且没有用户对话工具 |
| 指标 | `Rloose`、乘法瓶颈 `Rstrict`、精确全满足 `Rsucc` | Reward v3 的终局类型；正式 strict success 只认合法 `gold_purchase` |
| Agent prompt | 通用的收集信息、环境交互、流程控制 | 明确写出搜索、候选、证据、规格、购买、终止的决策规则 |
| 运行时 | 论文动作格式与环境交互 | JSON Schema 工具 + 当前页合法性约束 + Action Guard + observation projection |
| 论文训练规模 | Qwen3-8B；6K 条 GPT-4.1 成功轨迹；4 epochs | Qwen3.5-2B；379 train + 49 validation；成功轨迹筛选；3 epochs |

论文明确说明低总体成功率主要来自细粒度 attribute/option grounding、长程多轮澄清和个性化；Qwen3-8B 的 `Rsucc` 从单轮 `14.13%` 降到多轮 `6.48%`。当前项目拿掉了用户澄清与个性化这两个原论文难点，同时增加了更强的执行脚手架。因此当前 SFT 的较高分数首先证明“这个**系统**在当前单轮协议上可学”，不能反推出原环境整体容易。[ShopSimulator Introduction、§3.2](https://arxiv.org/pdf/2601.18225)

另一方面，当前 strict success 仍要求环境正常结束、`reward_valid=true`、`purchase_success=true` 和 `gold_purchase`，并非“只要买了就算成功”。所以也不能仅凭 60.5% 就断言 Reward 放水。[Reward v3 文档](../../docs/reward-v3.md)；[仓库 contract](../../AGENTS.md)

### 1.2 当前增益中，“会走流程”占了很大部分

冻结 200 题结果：

| 模型 | Done | Strict success | Mean reward | Guard rejection |
|---|---:|---:|---:|---:|
| Base | 18.0% | 0.0% | -0.1105 | 752 |
| SFT | 96.5% | 60.5% | 0.4729 | 52 |
| GRPO step 100 | 96.5% | 62.0% | 0.5158 | 38 |

SFT 的主要作用很像原论文所说的 workflow prior：先学会搜索、打开、核验、选规格和合法终止。原论文自己的 Qwen3-8B 单轮实验也从 Base `14.13%` 提升到 SFT `32.47%`，并不支持“SFT 不该有大提升”；论文进一步报告 SFT+RL 的单轮 `38.89%`，多轮 `35.50%`。[ShopSimulator Table 3](https://arxiv.org/pdf/2601.18225)

### 1.3 “环境/Reward 是否太简单”应怎样证伪

不需要 GPU，先做四个静态或环境侧审计：

1. **检索捷径**：用任务原始 Query 直接跑确定性搜索，记录 gold/可接受商品第一次出现的 rank、首屏 recall、需要几次改写。若大多数 gold 首搜 top-k 命中，任务确实偏检索捷径。
2. **文本泄漏/高重合**：统计 Query 与 gold title/model/options 的 n-gram 重合、唯一型号命中率、搜索词是否近似商品标题。若成功高度集中在高重合任务，成功更像字符串定位而非长程决策。
3. **约束复杂度**：按 active category/budget/brand/model/function/option 数、规格轴数、近似候选数、目标首现页数分桶报告成功率。若 SFT 只在低约束、低歧义桶高分，不能称任务整体容易。
4. **脚手架贡献**：未来只需做最小 prompt/guard ablation，而不是重训。固定模型，比较完整 prompt、删去决策规则的简化 prompt，以及只记录但不拦截的 guard；若成功率大幅下降，增益属于 model+harness 系统，而不是裸模型能力。

## 2. 为什么 GRPO 只表现出小幅净增益

### 2.1 先把“提升小”与“证据弱”分开

当前 final-200 每题每模型只有一次 rollout。SFT→GRPO 的 21 个状态变化中，12 个改善、9 个回退，exact McNemar `p≈0.664`；数据没有足够功效确认 1.5pp。应保留两个同时成立的描述：

- strict success 的净变化很小、统计证据不足；
- Reward 与若干行为指标方向一致地改善（平均 Reward `+0.0429`、Guard `52→38`、repeat loop `27→25`、max step `5→3`）。

在没有多次 stochastic rollout 或更多独立测试题前，不应据此选择复杂算法。

### 2.2 更可能先限制 GRPO 的，是 group 信息量

当前每个 prompt 只采样 `n=4`，初始 SFT 已有 60.5% 单次成功率，并且 SFT 可能产生高度相似动作序列。GRPO 只有同题 rollout 之间 Reward 有差异时才有相对优势；全对、全错或同分 group 都几乎没有信号。项目已经实现最多补采三批的 dynamic sampling，但这不保证最终有效 batch 足够大。[当前 GRPO 配置](../../configs/grpo.yaml)；[dynamic sampling 文档](../../docs/grpo.md)

应优先从现有/下次日志记录：

- 每 step 的 zero-variance group 比例、Reward unique count、group reward std；
- `num_gen_batches`、跳过 update 次数、每 update 的有效 prompt/trajectory/token 数；
- 同题 4 条 rollout 的唯一工具序列数、第一处分叉 Turn、搜索 query 唯一数、轨迹 pairwise edit distance；
- 成功率分桶：全成、混合、全败，并跟踪它们随 step 的迁移。

若 zero-variance 很低而梯度仍弱，dynamic sampling 不是主因；若很高，则先扩大同题探索或调整任务采样，而不是先加逐 Turn 算法。

### 2.3 SFT 过拟合/低熵假设目前缺哪块证据

当前训练配置关闭 entropy 计算，因此“低熵”尚未被观测。success-only Teacher 数据还可能造成两种不同现象：

- **好现象**：动作协议稳定，减少无效探索；
- **坏现象**：搜索表达、候选比较和失败恢复模式过窄，同题 rollout 几乎相同。

最干净的可证伪设计不是笼统地换一个较差 SFT，而是比较 SFT 的早/中/晚 checkpoint：

| 指标 | 若“训过头”成立，应看到 |
|---|---|
| SFT validation action-NLL | 继续下降或持平 |
| frozen task 成功率 | 已饱和 |
| 同题 rollout 工具序列/查询多样性 | 随 checkpoint 下降 |
| token/action entropy | 随 checkpoint 下降 |
| GRPO zero-variance group 比例 | 随 checkpoint 上升 |
| 固定少量 GRPO updates 后的增益 | 晚 checkpoint 小于中 checkpoint |

只有最后一行才能直接支持“可塑性下降”；单看初始熵低不够，因为低熵也可能只是合法动作更确定。

### 2.4 失败是否可由 policy 修复

SFT 的 79 个非严格成功样本包含 32 个 partial alternative、27 个 repeat loop、7 个 wrong purchase、5 个 max steps、1 个 reward unverifiable 和 7 个 unknown。下一步最值钱的是将每条失败按“最早可归因 Turn”分类：

1. gold/可接受候选从未出现在可见 observation：检索或环境覆盖问题，policy credit 很难修；
2. 候选出现但被忽略/放弃：搜索与候选利用问题，适合 RL；
3. 打开商品但未核验关键字段：证据核验问题，适合 Turn credit；
4. 核验正确但选错规格/价格：决策问题，Reward 应聚焦弱维度；
5. 重复、无进展、max-step：终止/恢复问题，已有确定性事件可定位；
6. infrastructure invalid / reward unverifiable：应过滤，不应当作负样本学习。

特别应人工或规则审阅 12 个 SFT fail→GRPO success 与 9 个 success→fail 的配对轨迹。净增只有 3 时，解释这 21 条比再看 aggregate mean 更有信息。

## 3. DAPO：哪些机制相关，哪些已经存在

DAPO 是为 Qwen2.5-32B base 的长 CoT 数学 RL 提出的系统。论文消融从 naive GRPO 的 AIME avg@32 `30`，依次加入 overlong filtering、Clip-Higher、soft overlong punishment、token-level loss 和 dynamic sampling，最终到 `50`；这是组合消融，不能把每一项的数值原样外推到多轮购物 Agent。[DAPO §4.2、Table 1](https://arxiv.org/pdf/2503.14476)

尤其要注意：DAPO 主实验**不是 SFT→RL**，而是从 Qwen2.5-32B Base 开始。因此下面对“SFT 后低熵”的联系属于机制推断，不是论文已验证的结论。四项技术也针对四类不同故障，不能合称为“低熵修复包”。

| DAPO 机制 | 原论文作用 | 当前项目状态 | 对本项目的判断 |
|---|---|---|---|
| Clip-Higher | 将 PPO 下/上界解耦；论文用 `εlow=0.2, εhigh=0.28`，给低概率探索 token 更大上升空间，缓解 RL 过程 entropy collapse | `clip_ratio_low=0.20`、`clip_ratio_high=0.20` | 候选消融；但只在 upper-clip fraction 高、熵/多样性随 RL 降低时有明确靶点。它不能自动恢复一个已冻结 SFT 中不存在的行为模式 |
| Dynamic Sampling | 过滤同题全对/全错 group，补采到有效 batch；官方 recipe 是 `n=16`、最多 10 个 generation batch | 已启用；`n=4`、最多补采 3 批、连续跳过有上限 | 已有最小版本；先量 zero-variance、补采和跳更次数，再决定是否扩大 |
| Token-level Policy Gradient | 全 batch token 归一化，避免“先序列内平均、再序列平均”导致长样本每 token 权重过低 | `loss_agg_mode: token-mean` | 已有，不应重复实现。要核对 veRL 0.8 实际路径确实使用该 aggregation |
| Overlong Filtering | 对截断样本 mask loss，避免“正确但太长”被错误惩罚 | context 超限会标 infrastructure invalid；另有环境终局 `max_steps` | 原则适用：基础设施/上下文截断应过滤，不应混为任务失败 |
| Soft Overlong Punishment | 只在最大 token 长度前的一段 buffer 内线性惩罚，论文为 16,384 后的 4,096 token buffer | 项目是按**环境动作步数**的可选 penalty，默认关闭，且 Reward v3 已有 max-step/repeat outcome | 两者不是同一机制。不要直接打开以“复现 DAPO”；先判断失败来自 token 截断还是无效动作，否则会重复惩罚长但必要的轨迹 |

DAPO 官方代码也明确配置 `clip 0.2/0.28`、`token-mean`、group filtering、overlong buffer；这与上表相符。固定复现版本中，dynamic sampling 按 prompt `uid` 聚合 metric，只保留标准差大于零的 group，未凑满 train batch 才继续生成；它不会给全同分 group “造梯度”。[DAPO 官方仓库](https://github.com/BytedTsinghua-SIA/DAPO)；[veRL 固定复现 recipe](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_qwen2.5_32b.sh)；[dynamic sampling 实现](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/dapo_ray_trainer.py#L158-L205)

### Clip-Higher 的决策门槛

先记录三条曲线：generation entropy、mean upper-clipped probability/upper clip fraction、同题 rollout 多样性。若三者分别表现为“熵下降、upper clipping 显著、多样性下降”，再只改 `clip_high` 做单变量消融；若 upper clipping 接近零，提高上界不会解决探索问题。DAPO 自己也是从这些训练曲线诊断，而不是把 0.28 当通用常数。[DAPO §3.1](https://arxiv.org/pdf/2503.14476)

Clip-Higher 只能放大**已经被 on-policy rollout 采到且获得正优势**的低概率 token；从未采到的行为仍没有梯度。若 SFT 后四条 rollout 几乎完全相同，先提高 rollout 多样性和有效 group 比例可能比改 clip 更直接。相反，token-level loss主要修正长短序列的 batch weighting，overlong shaping 主要修正截断 Reward noise，两者都不是初始增熵机制。[DAPO §3.1–§3.4](https://arxiv.org/pdf/2503.14476)

### 长度 shaping 的决策门槛

先分清三类长度：单个 assistant turn token、全轨迹 context token、环境 action steps。DAPO 处理前两者中的“生成被截断”；当前可选配置惩罚第三者。对购物 Agent，建议先：

1. mask/过滤 infrastructure-invalid 和真正截断样本；
2. 对 repeat/no-progress 使用现有确定性终止；
3. 只有“同样成功但存在无意义额外动作”的成对证据充足时，才加轻量 step penalty。

## 4. AgentOPSD：逐 Agent Turn 信用分配究竟怎么做

AgentOPSD v1 于 2026-08-06 提交。它解决的对象正是 GRPO 将同一个 trajectory advantage 广播给所有 turn 的问题。[arXiv 记录](https://arxiv.org/abs/2608.05987)

### 4.1 算法信号

对 student 已经 on-policy 采样的第 `k` 个动作 turn `a_k`：

1. 同一参数 `θ` 做两次 teacher-forcing 打分：普通上下文 `s_k`，以及增加训练期 privileged skill `c+` 的上下文 `(s_k, c+)`。
2. 每个动作 token 的 detached gap 为 `δ_{k,t}=log πθ(y_{k,t}|s_k,c+)−log πθ(y_{k,t}|s_k)`；turn evidence 是 `e_k=Σ_t δ_{k,t}`。
3. 以同题 group 的二元成功率 `B_0=S/G` 作为 prior，在 log-odds 空间递归：`c_k=γc_{k−1}+e_k`，`B_k=sigmoid(logit(B_0)+c_k)`。
4. `ΔB_k=B_k−B_{k−1}` 表示该 Turn 对累计成功 belief 的边际修订；再用最终 sequence advantage 的符号对齐：`q_k=sign(A_seq)ΔB_k`。
5. 在轨迹内标准化 `q_k`，变成有界 multiplier，并与原 `A_seq` 混合；该 turn 的所有 response token 继承这个 reshaped advantage。

因此，它不是独立的 process reward model，也不额外 rollout；成本是每条轨迹一次 skill-conditioned teacher forward，belief block 没有参数。论文实验使用 binary verifier、group size 8、`λ=0.5`、multiplier band `b=0.2`、`γ=0.95`、clip `0.2/0.24`。[AgentOPSD §2、Appendix B/F](https://arxiv.org/pdf/2608.05987)

这里的 Bayesian 解释有明确假设：skill-conditioned 分支应近似 success-conditional action distribution；在更强的推导中还假设成功较稀少，使未加 skill 的混合策略主要由失败分布主导。只有前一假设时，`e_k` 更接近 action 与成功事件的 pointwise mutual information。论文也明确说 `B_k` 是 relative support，不是校准后的真实成功概率。因此 `ΔB_k` 是有理论动机的 evidence proxy，不能解释成该 Turn 的已识别真实因果效应。[AgentOPSD §2.2、Appendix A](https://arxiv.org/pdf/2608.05987)

### 4.2 它为何不能直接成为当前 Reward v3 的一个开关

必要条件与当前差距：

- **privileged skill**：论文从 SkillRL 的 SkillBank 按关键词检索训练期 skill，推理时移除。当前项目的丰富 system prompt 对 student 也可见，不能产生 teacher-student gap。
- **binary prior**：论文的 `B0=S/G` 来自 `{0,1}` outcome。本项目 Reward v3 是多值终局 utility。可以把 `gold_purchase` 作为 binary verifier，但这属于需要验证的迁移，不是论文已经证明的设置。
- **turn 对齐**：当前 response mask 和工具边界理论上足够定义 turn，这是最匹配的一部分。
- **计算与实现**：论文实验是 3B/7B、8×H800，WebShop 最多 15 turns；本项目最长 35 个环境动作、20K response budget。论文官方仓库截至 2026-08-10 仍未发布实现。
- **避免答案泄漏**：不能把 gold ASIN、隐藏目标字段直接作为 `c+`。更安全的是只从 training task 的失败类别提取通用 skill，例如“打开候选后核验预算与全部 option，再决定购买”，且确保不含任务答案。

最小可行研究顺序应是：先验证失败是否集中在少数 pivotal turns；再构建不含答案的通用 skill bank；最后才比较 vanilla GRPO 与 turn reshaping。否则实现 AgentOPSD 只会把未知的 skill 质量引入新的混杂变量。

论文的结果说明这一方向值得研究，但不应高估外推性：Qwen2.5-7B 上，AgentOPSD 相比 GRPO 在 ALFWorld 为 `89.1 vs 81.2`，Search-QA 为 `49.2 vs 42.0`，WebShop exact completion 为 `79.7 vs 72.6`；所有对比均共享其 SkillBank、二元 verifier、group size 8 和训练预算。[AgentOPSD Table 1、§3.1](https://arxiv.org/pdf/2608.05987)

### 4.3 对当前任务最有价值的 AgentOPSD 可证伪前提

利用已有轨迹做 CPU 分析即可：对每条失败标注“最早不可逆错误 Turn”，再看错误 Turn 是否集中。如果大量失败能定位到少数 `open_product / select_option / buy_now / finish` 决策，而其前后是例行合法操作，逐 Turn credit 有明确价值；如果多数失败是“目标从未被检索到”或全程信息不足，credit reshaping 不会创造缺失的候选与证据。

## 5. 无 GPU 的诊断清单（按信息价值排序）

### P0：现有结果即可完成

1. 输出 200 题 SFT↔GRPO 配对转移矩阵和 exact McNemar，而不是只报两个边际成功率。
2. 对 12 个改善、9 个回退逐条比较最早分叉 Turn、Reward type、candidate visibility、guard/repeat/max-step。
3. 将全部 SFT 失败按“检索不可达 / 候选忽略 / 证据不足 / 规格或购买错误 / 循环终止 / 基础设施无效”互斥归因。
4. 分别报告 `P(success|done)`、`P(success|reward_valid)`、完成率、合法性和商品约束满足，防止协议能力掩盖决策能力。

### P1：只需环境和数据，不需模型

1. Query→gold 的 deterministic search rank/top-k recall。
2. Query/gold title/model/options 的文本重合和唯一标识泄漏。
3. active constraints、规格轴、近似候选数、目标首现页数的难度分布。
4. 新旧 SFT 数据的任务覆盖、长度、工具序列、搜索 query 和失败恢复策略多样性；成功数更多不等于行为支持更广。
5. train/SFT/GRPO/eval 的 task_id、近重复 Query、目标商品/品类/型号交叉污染检查。

### P2：需要已有训练/rollout 日志，但不需 GPU

1. 每 step 的 group reward std/unique count/zero-variance ratio、dynamic resampling 次数和 skipped update。
2. policy entropy、mean token probability、upper/lower clip fraction、approx KL、grad norm、response/action length。
3. 同题 rollout 的动作序列、搜索 query、访问商品和终局多样性。
4. 每种失败类型获得的非零 advantage token 数，检查 Reward 是否真的给短板提供梯度。
5. checkpoint 级 paired eval，而不是用训练 Reward 选 checkpoint。

## 6. 建议的最小研究路线

1. **先做失败图谱和 group 信息量审计**。这能区分环境/检索瓶颈、Reward 区分度不足、以及 SFT 低多样性。
2. **再做 SFT checkpoint 可塑性对照**。不要以“较差 SFT”作为目标，而应找“协议已学会、尚有探索多样性”的 checkpoint。
3. **若 upper clipping 与 entropy collapse 同时出现，只消融 Clip-Higher**。本项目已经有 token-mean 和 dynamic sampling，避免把 DAPO 全部重做。
4. **长度先过滤截断噪声，不先惩罚长轨迹**。只有确认无意义动作导致长度增长时才打开轻量 step shaping。
5. **只有 pivotal-turn 假设被现有失败轨迹支持时，再研究 AgentOPSD**；并先解决 leakage-safe skill bank 与二元 verifier 定义。

## 一手来源

- [ShopSimulator 论文（arXiv:2601.18225）](https://arxiv.org/abs/2601.18225)
- [ShopSimulator 官方仓库](https://github.com/ShopAgent-Team/ShopSimulator)
- [DAPO 论文（arXiv:2503.14476）](https://arxiv.org/abs/2503.14476)
- [DAPO 官方仓库](https://github.com/BytedTsinghua-SIA/DAPO)
- [veRL 固定 commit 的 DAPO recipe](https://github.com/verl-project/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo)
- [veRL 中 Clip-Higher 与 token-mean 实现](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/trainer/ppo/core_algos.py#L246-L331)
- [veRL 中 DAPO overlong Reward 实现](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/workers/reward_manager/dapo.py#L90-L101)
- [AgentOPSD 论文（arXiv:2608.05987）](https://arxiv.org/abs/2608.05987)
- [AgentOPSD 官方仓库](https://github.com/ZethWang/AgentOPSD)
