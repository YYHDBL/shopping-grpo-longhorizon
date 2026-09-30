# V2.0 项目工作总结

> 日期：2026-09-27。范围：V2.0 设计讨论（2026-09-23 起）至 SFT 训练就绪。
> 详细决策依据见 `2026-09-23-v2-design-baseline.md`，V1 问题对照见 `2026-09-23-v2-update-notes.md`，数据验收记录见 `2026-09-25-data-acceptance.md`。

## 一、项目目标

在 ShopSimulator 环境上构建规模可控、流程完整、可复现、可观测的购物 Agent 开源项目：数据验收 → 画像与切分 → 双 Teacher 教材采集 → SFT → 在线 GRPO → 冻结评测，形成 bad-case 数据飞轮。技术栈锁定 vLLM 0.25.1 + veRL 0.8.0 + Qwen3.5-9B（后训练版），双 A800 全参 BF16 FSDP。

## 二、各阶段成果

### 1. 设计对齐（14 项用户决策）

画像构造、Query–Gold 两级验收、dev 1,050、不补数据、Teacher 采集参数、SFT 数据契约（全链路关思考、loss 只算动作）、评测报告结构（完成率 + 难度/画像/领域分桶 + 四类错误归因）、Jev 校准流程、Reward 数值表（v3 为基，改两处）、RL 任务池与 SFT 隔离、三档对照实验、Harness 三项（required + 删 think、子页单次拦截、预算分级截断）、提示词分层（Student 仅协议层 / Teacher 含决策规则）、双 Teacher 分治。

### 2. 数据验收（源 23,421 条 → 可用 19,106 条）

- 两级验收：本地硬约束（规格 501 缺失、超预算 453）+ Jev 全量语义（22,467 次调用，$0.70）；
- 复查：六个轻量 subagent 盲判发现 Jev 偏严（价格容差、同义表述），救回 2,475 条、剔除 39 条异常定价；
- 最终：accepted 19,106（81.5%）/ semantic_fail 3,310 / hard_fail 954 / bad_pricing 39 / unverifiable 12。

### 3. 画像与切分

- 画像池 4,666 份（773 份搜索关键词泄漏标记），渲染层统一剔除行为清单字段；
- 三条件分配（最终 52/24/24，353 条三轮重配失败降级无画像）；配对复核 9,554 条（subagent 4,961 + Jev 4,595）；
- dev 精确 1,050 条（难度×领域×条件 81 格分层）。

### 4. Harness 修改（代码完成，143 测试全过）

tool_choice=required（删纯文本出口）、删 think 工具、子页单次访问拦截、observation 预算 3072/6144/1024、并行工具调用统一为丢弃多余、按钮清单与预算单一来源。环境与工具集一行未动。

### 5. Teacher 教材采集（6,579 题 → 5,852 条教材）

- 双 Teacher 分治：GLM-5.3-Flash 与 DeepSeek V4.1 Flash 各半分片、各 8 并发、环境 16 槽；
- 每题预算 3 次成功即停；GLM 通过率 90.4%、DeepSeek 87.5%（收敛提示词使 DeepSeek 从 44% 提升至 87%+）；
- 教材构成：hard 占 34.6%、平均 5.9 步；teacher-hard 队列 727 题；全轨迹留存；
- 成本：输入 4.78 亿 + 缓存命中 5.33 亿 + 输出 871 万 token。

### 6. Jev 校准与冻结（v3）

50 条分层考卷 + DeepSeek 盲判裁决：一致率 88%（过线）、满分精确率 93%。据此修订：价格低于预算不算违规、字段沉默不单独扣死。v3 为训练评测期间冻结版本。

### 7. SFT 教材加工与训练格式

- `samples.jsonl`：Student 版 prompt（动作来自 Teacher、提示词移植即蒸馏）+ 逐消息 loss mask（仅 assistant 训练）；
- `train.parquet`：veRL MultiTurnSFTDataset 格式（messages + tools + enable_thinking，5,852 行），显式 arrow schema 写入，读回 + Qwen3.5 模板渲染抽验通过。

### 8. 训练环境

- veRL 0.8.0 + vllm 0.25.1 + torch 2.11.0+cu130 装入 `.venv`（uv 解依赖），CUDA 可用；
- 模型就位：`models/Qwen3.5-9B/`（19G）。

## 三、执行中的关键经验

1. 大批量语义判断优先 Jev API（分钟级、$0.1 量级）；subagent 并行易触发账户限流且耗时不可控；
2. 服务器三个模型端点均可直连，全部客户端已代码级绕过代理（ProxyHandler({})），不依赖 SSH 隧道；出站故障要用全局熔断保护任务队列；
3. 环境 tmux 内 Ctrl+C 会杀采集进程；退出 tmux 用 Ctrl+B 再 D；
4. 根分区常满：pip/uv 的 TMPDIR 必须指向 /data；大依赖安装用 uv（pip 解不开 vllm+verl 的 numpy 冲突）+ 清华镜像 + 循环重试；
5. arrow 无法写入无字段的 struct：工具 schema 与调用参数的空对象一律省略键或用显式 schema。

## 四、当前状态与下一步

| 事项 | 状态 |
|---|---|
| 数据/画像/采集/校准/教材/格式 | 全部完成 |
| 冒烟（GPU）：token 边界实测、required 支持、单步前向 | 待用户指令 |
| SFT 训练 | 冒烟后 |
| GRPO / 三档评测 | SFT 后 |

基础设施现状：tmux 会话 `shopsim-env`（环境服务 5700 端口、16 槽）；关键数据：`outputs/acceptance/final_verdicts.jsonl`（数据定案）、`outputs/split/tasks_final.jsonl`（任务定义）、`outputs/collection/`（全部轨迹）、`outputs/sft_dataset/train.parquet`（训练数据）。

---

# 追记：2026-09-28，从训练格式到训练启动

## 五、训练格式审计与修复

外部审计发现七项问题（veRL 数据集实例化失败、arrow schema 的 None 参数污染、序列长度未定、教材准入契约、CUDA 不可用、测试失败、teacher 列空），由修复 Agent 在 worktree `.claude/worktrees/v2-fix` 完成，本会话逐项验证通过：

- **守卫拒绝段不入训练**：5,852 条教材中 941 个被拒 assistant 段（684 条轨迹）train=0，逐消息标记贯通 samples.jsonl → parquet → 数据集 loss mask；6 条 `__dummy__` 参数污染轨迹剔除；
- **审计字段落盘**：reward_valid/over/termination_reason/jev_verdict/purchase 全进 parquet，431 条替代购买可追溯；
- **fail fast**：非法 JSON、空参数值、token 统计缺失一律报错，禁止静默转换；
- 测试 158 过 1 跳；数据集 14 项在真实 tokenizer 下全过。

## 六、环境死锁与容器路线（重要转折）

**死锁链**：Qwen3.5 需要新 vllm（0.25 系）→ 绑定 torch 2.11（仅 cu130 构建）→ 宿主机驱动 535 跑不了 cu130；而能跑的 vllm 0.12.0 不认 Qwen3.5。宿主机无解。

**解法**：服务器上在跑的推理服务全用 docker 容器（镜像 `d5219758abb3` 内带 CUDA 兼容层 cuda-compat，绕开驱动限制）。我们照做：

- 起自己的容器（挂载 /data、GPU 直通），容器内栈：**verl 0.9.1 + vllm 0.25.0（认 Qwen3.5）+ torch 2.11+cu130 + transformers 5.10.4，Python 3.12**；
- 基线版本锁变更（用户批准）：verl 0.8.0→0.9.1、vllm 0.25.1→0.25.0、transformers <5.11；
- 装法：pip 装 verl 不带 vllm extra（否则它 pin vllm==0.24.0 替换镜像的 0.25.0）；`--entrypoint sleep` 覆盖镜像入口；unset 代理 + 清华镜像；
- 宿主机 `.venv`（torch 2.9+cu128 组合）保留做 CPU 数据处理，**不认 Qwen3.5，训练一律走容器**。

## 七、GPU 冒烟（通过）

用户授权临时停其 `gemma-4-31B-ita` 服务腾出 4-7 卡：容器栈 CUDA 真实跑通（4×A800 矩阵乘）、Qwen3.5 transformers 前向生成（8.95B）、vLLM 0.25.0 引擎加载生成全通过。冒烟后服务已恢复。卡位地图：0-1 gemma-it、2-3 JYHLLM、4-7 gemma-ita（均用户自有）。

## 八、SFT 启动准备（当前状态）

- 训练数据：train 5,542 + val 110（2% loss 验证集，固定种子）；
- train_sft.sh 按 verl 0.9.1 配置结构重写（engine/optim 分组、val_files、test_freq=50），hydra dry-run 逐字段验证落位；模型软链指向主目录；
- GPU 容器 `shopping-gpu` 在跑（透传 4-7 卡，容器内编号 0-3）；
- 用户定下并行方案：物理 6-7 卡跑 SFT（容器内 CUDA_VISIBLE_DEVICES=2,3）、物理 4-5 卡跑 Base 基线评测（evaluate_model.py，冻结 eval 1,092 条）；
- **当前卡点**：SFT 启动报 FlashAttention2 未安装（verl 0.9.1 FSDP 默认开）。flash-attn 无预编译包不装，正解是配置改 SDPA 注意力——已移交新 Agent 处理（提示词已交）；
- SFT 预计时长 2~3 小时（双 A800、BF16、动态 batch）；Base 评测预计半天内出数。

## 九、追加经验

6. `torch.cuda.is_available()` 会返回假 True（驱动过旧时真正初始化才报错），CUDA 验证必须真做 GPU 计算；
7. 容器内 `CUDA_VISIBLE_DEVICES` 用容器本地编号（透传后重编号），不是宿主机编号；
8. verl 大版本间配置结构会变（0.8→0.9 的 model/engine/optim 从平铺变分组），升级后必须 hydra dry-run 重新核对字段；
9. 三档对照的第一档（原模型零样本基线）必须在 SFT 前跑掉，否则提升幅度无从对比——本次差点跳过，用户拦住了。

---

# 追记：2026-09-29，GRPO smoke 验证完成

## 十、GRPO smoke 全链路验证

外援修复（`PYTORCH_NVML_BASED_CUDA_CHECK=1`，兼容 hook 中的 CUDA 过早初始化导致 Ray worker GPU 映射缓存错误）后，GRPO 多卡 FSDP 训练在 Docker 容器内全链路跑通：Ray 集群 → FSDP 4 卡 → vLLM colocate → Agent rollout（环境交互 + 工具调用）→ Reward（含 Jev 灰区判定）→ GRPO 梯度更新 → SwanLab 指标记录。连续 3 步正常，指标健康（reward 0.767、entropy 0.38、clip_ratio 0、489s/步）。

## 十一、GRPO 正式跑配置（已验证的安全值）

| 参数 | 值 | 备注 |
|---|---|---|
| colocate 4 卡 | hybrid_engine=true | 训练与 rollout 分时复用 |
| group size | 8 | 每题采 8 条做组内比较 |
| train_batch_size | 8 题/步 | ×8 采样 = 64 条轨迹/步 |
| vLLM gpu_mem_util | 0.35 | 0.5 会 OOM (wake)；0.45 可试 |
| ppo_mini_batch | 8 | 16 会 OOM (log_softmax) |
| ppo_micro_batch | 1/GPU（正式跑可试 2）| |
| ppo_max_token_len | 24576 | 16384 会截断合法序列（18k+）|
| optimizer_offload | true | 正式跑必开，解决跨步显存累积 |
| RL 数据量 | 1000 题 | 对齐 V1，约 125 步 ~17h |
| 环境 slots | 80 | 64 rollout 并发零错误 |

## 十二、GRPO 算法配置确认（无花活）

- KL loss: off / KL in reward: off / KL ref model: disable（连参考模型都不加载）
- Entropy bonus: 0（纯记录，不影响 loss）
- Loss: ppo_clip / Advantage: grpo / Advantage std 归一化: off
- 唯一正则化是 PPO clip 0.2

## 十三、关键指标监控清单

reward/mean（应升）、reward/std（应降）、actor/entropy（缓慢降，勿塌到 0）、
actor/pg_loss、actor/grad_norm、actor/clip_ratio（<10%）、response_length/mean、
success_rate（gold）

## 十四、权重清理

- 删除：GRPO smoke checkpoint（106G）+ SFT 中间步 100/150（~105G）
- 保留：SFT 最终 step_172（53G，正式评测用）+ sft-merged（18G，GRPO 起点）+ Qwen3.5-9B 原始（19G）

---

# 追记：2026-09-29 下午，run1 三版排障到分块输出头定稿

## 十五、正式跑三次 OOM/重启全程

| 版本 | 配置 | 结果 |
|---|---|---|
| v1 | optimizer_offload=true，gpu_mem 0.35 | step1 过（峰值 60.9G），**step2 死于 log_softmax 的 logits 峰值** |
| v2 | +param_offload=true，gpu_mem 0.30 | step1 峰值 56.8G；用户暂停对齐，未验证 step2 |
| v3 | **+分块输出头（外援方案）** | step1 峰值 **43.8G**，update_actor 反而快 77s，步时 446s（全程约 15.5h） |

崩溃点：`update_actor → forward_step → logprobs_from_logits_v2 → F.log_softmax`，
GRPO 训练前向物化 [seq, vocab] logits（22k × 15 万词表 bf16 ≈ 6.7G/张量）+ log_softmax
中间结果 + 反向，把 80G 卡顶穿。reserved-unallocated 仅 283MB，非碎片化。

## 十六、外援方案（已逐条对源码核实后采纳）

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: true
    fused_kernel_options:
      impl_backend: torch
    enable_activation_offload: true
```

原理：veRL 0.9.1 `FusedLinearForPPO`（utils/experimental/torch_functional.py）以
512 token/块直接用 hidden_states × lm_head.weight 算 log_probs+entropy，反向逐块
重算，**从不物化 [seq,vocab] logits**。Qwen3.5 分发在 monkey_patch.py:270。生效
标志：初始化日志 `Using Torch backend for fused kernels in Qwen3_5ForConditionalGeneration`。

## 十七、本轮经验（按教训价值排序）

1. **方案排除要说"我没找到"，不要说"不存在"**：我断言压峰值只有"减序列/offload/
   砍上限"三条路，漏查了 `use_fused_kernels`（藏在 experimental 目录）。外援直接
   给出第四条。自包含问题总结（环境/栈/崩溃栈/已排除项）请外援的模式再次证明有效。
2. **改配置前先翻旧版本**：V1 的 GRPO 配置里 param_offload 早就是 true，我照抄了
   smoke 验证配置（false，3 步短轨迹没暴露）。
3. **liger FLCE 为什么救不了 RL**：RL 需要逐 token log_prob 算新旧策略比值，必须
   物化 logits；FLCE 只吐标量 loss，veRL 源码写死 `fused_linear_cross_entropy=False`
   关掉它。SFT 只需总 loss 所以能用（V1 的 liger 开关在 SFT 管线）。正解是分块输出
   头——**分块必须发生在 lm_head 之前且反向重算**，只对已物化 logits 分块调
   log_softmax 没有用。
4. **动态 bsz 语义**：use_dynamic_bsz=true 时 microbatch 按 token 预算（24576）打包，
   micro_batch_size_per_gpu=1 不保证单条轨迹。预算已等于最长单条序列需求，降预算
   降不了单条 22k 轨迹本身的计算量。
5. **截断奖励的坑**：截断轨迹（error=assistant_finished_without_environment_done）
   的 reward_version 还是 None（环境没来得及报终局），reward 分支要在 v3/兜底分支
   之前统一拦截，否则改了 v3 分支也不触发。
6. **容器内 nvidia-smi 的 PID 是宿主机 PID**，容器 /proc 查不到属正常；容器
   NVIDIA_VISIBLE_DEVICES=all 时看到的 0-7 就是物理 0-7，别把自己的卡位认错。
7. **colocate 不是"一张卡同时跑两套"**：是分时独占，gpu_memory_utilization 调的是
   vLLM 显存配额（不是计算利用率），压它牺牲 rollout 并发换 wake 阶段不挤兑。

## 十八、reward v3.1 定稿（含一次否决）

- 截断（token/步数预算耗尽未终局）：0 → **-0.5**，堵"拖满预算 0 分 > 礼貌收尾
  -0.15"的漏洞；基础设施无效仍 0.0（不制造学习信号）
- 连续长度惩罚：实现后**被否决撤销**（用户裁决：目标是预算内完成任务，长轨迹可能
  包含必要的搜索/比较/规格确认，仅凭步数无法判断哪些是浪费）。turns 留在
  extra_fields 做纯观测
- 完整口径：gold/Jev-fully 1.0 / partial 0.25 / 礼貌停 -0.15 / 早退 -0.35 /
  截断 -0.5 / 买错 -0.85 / 基础设施无效 0.0
- Jev 灰区成本实测口径：1000 题 125 步约 300~600 次调用，$0.02 量级，可忽略

---

# 追记：2026-09-30 凌晨，run1 熵爆炸中止与打捞

## 十九、run1 v3 训练到 step93 主动中止：熵爆炸

**现象**（数据全部来自 step 指标行）：

- 熵：step1-41 稳定 0.36~0.48 → step45 起爬升（0.64）→ step55 加速 →
  step75 1.85 → step87 **6.03** → step90 5.75（爆炸，采样分布接近随机）
- 验证（64 题贪婪）：step0 0.671 → **step25 0.490（掉坑）** → step50 0.698 →
  step75 0.697（贪婪性能存活——argmax 对适度平坦不敏感，但采样已废）
- 训练 reward 剧烈震荡不收敛；step91 批均值 **-0.17**（采样大面积失败）
- grad_norm 0.87~0.97 正常、轨迹长度稳定 4.5~7k、clipfrac 恒 0（正常）

**诊断**：无 KL 锚 GRPO 的策略熵爆炸（不是塌缩）。组内信号全程存活
（优势范围始终 ±1~1.5，从没塌 0），不是"没信号漂移"，而是信号大 +
全无刹车（no-KL、无熵约束、无 std 归一化）+ lr 1e-6 持续单向推。
恶性循环：熵涨 → 采样更随机 → 失败率涨 → 组内分差更大（优势 max 从
0.69 涨到 1.51）→ 推力更强 → 熵更涨。

**打捞**：checkpoints 25/50/75 在。选 **step50** 做正式评测（val 0.698 与 75 持平，
但熵 0.48 在健康区、75 的 1.85 已进发散段）。

**下一轮必改清单**：

1. 记组级指标：agent_loop 结算处算每题 8 条的组内 mean/std 进 metrics
   （veRL 无现成组级指标，本次只有优势 min/max 代理）
2. 开 `trainer.rollout_data_dir` 落盘每条轨迹 reward，可离线复盘
3. 加刹车（三选一或组合）：KL 锚（ref model，显存现在装得下）、
   负 entropy 系数、lr 减半 + cosine 衰减
4. 早期预警线：熵连续 10 步 > 0.8 或验证单次掉 >10pp 即人工介入

**运维教训**：colocate 崩溃后清显存要连 `VLLM::EngineCore` 一起杀
（不是只有 VLLM::Worker）；GRPO checkpoint 目录比 SFT 多一层 `actor/`，
合并命令 local_dir 要指到 `global_step_N/actor`。

## 二十、本次评测安排

- 合并：`verl.model_merger merge --backend fsdp --local_dir checkpoints/grpo_run1/
  global_step_50/actor --target_dir outputs/models/grpo-50-merged`（18G）
- 评测：与 SFT/base 同管线（冻结 1092 题、贪婪、8 并发、vLLM 8000 + 环境 5700），
  之后 evaluate_jev.py 灰区判定，与 SFT 60.7% / base 39.2% 对比

## 二十一、run1 评测结果：三臂定稿，GRPO 本轮无收益

| 臂 | 严格完成率 | 真实完成率（Jev） |
|---|---|---|
| Base | 39.2% | 52.8% |
| SFT | 60.7% | 72.1% |
| **GRPO run1 step50** | **57.4%（627）** | **70.3%（740/1052）** |

- Jev 灰区：247 条替代购买中 113 条升级 fully（46%，与 SFT 时代校准一致）
- 退化形态：gold -36 → repeat_loop +33（策略变抖），顶格拖延反而 -15
- 评测耗时 50 分钟（SFT 同管线 62 分钟，速率同量级）
- 定性：熵爆炸前的 step50 也已受损——val@25 掉到 0.490 说明发散从 warmup 后
  不久就开始了，"选未发散 checkpoint"救不回已发生的学习污染
- 处置：vLLM 服务已停、卡清空；等待外援经验配置起 run2
- 运维补充：评测 vLLM 必须带 `--enable-auto-tool-choice --tool-call-parser
  qwen3_coder`（照抄 scripts/serve_model.sh），漏了第一通 tool_choice 请求就 400

---

# 追记：2026-09-30，run2 筹备（外援刹车组合）

## 二十二、run2 配置定稿与核实记录

外援方案逐项对源码核实后采纳：KL 锚（use_kl_loss=true, 0.01, low_var_kl）+
lr 2e-7（warmup 8 步 → cosine → min_lr_ratio 0.1）+ batch 16×n8=128 轨迹 +
top_p 1.0 + rollout IS 修正（token, 阈值 2.0）+ filter_groups 显式关。
完整配置 `configs/grpo_run2.yaml`（独立文件不继承，避免 hydra 合并惊喜）。

**核实中的发现**：

1. **`algorithm.disable_kl` 是死字段**：0.9.1 全源码无读取点，run1 里那行
   true 是摆设；真正控制 ref 加载的是 `use_kl_loss` / `use_kl_in_reward`
2. **`policy_loss.loss_mode`（默认 vanilla）、`filter_groups`、
   `rollout_correction` 配置组都在**；lr 调度只有 constant|cosine 两选
3. **组内 std/零方差组比例 veRL 算了但不记**（core_algos.py 的 group_mean_std
   结果不进 metrics）——compat.py 里包装 `compute_data_metrics` 注入
   `group/reward_std_mean` 与 `group/zero_variance_ratio`，随原 dict 自动进
   console + SwanLab
4. **宿主机/容器 veRL 版本分裂坑**：组级补丁最初写进 hook 主体，宿主机
   .venv 的老版 veRL（DataProto 不在顶层）import 失败打挂单测——hook 拆成
   `install_torch_padding_fallback`（宿主机可测）与 `install_worker_hooks`
   （容器专用，配置指向后者）
5. **环境槽位规则**：并发轨迹数 = train_batch_size × n，必须 ≤ SHOPSIM_ENV_SLOTS。
   run1 8×8=64<80 幸运过关；run2 16×8=128>80 会产生抢槽失败的基础设施轨迹，
   已扩到 160（重启环境服务，RAM 913G 富余）

## 二十三、run2 冒烟结果（3 步全过，2026-09-30 04:10）

- [x] 分块输出头生效（actor + ref 双模型）
- [x] KL 锚工作：step3 kl_loss=4.06e-4 非零。**step1/2 的 0.0 是算术必然**：
  warmup 期 lr 仅 2.5e-8，bf16 下权重未挪动 → 策略与 ref 逐 bit 相同 → KL 恰为 0
- [x] 显存峰值 61.9G（ref + batch16 + IS 修正），余量 17G，零 OOM
- [x] 128 并发 rollout 零抢槽错误（160 槽）
- [x] rollout_data_dir 每步一个 jsonl，含 uid/score/完整轨迹
- [~] 组级指标：smoke 期间补丁经历两版修复后定稿——
  ① metrics_batch 不带 uid（TensorDict 重建），改两段式：包装优势计算函数
  暂存 + metrics 包装器注入；② **uid 真实结构是 `<task>_<sample>_<attempt>`**
  （中间段 0..7 递增，实测落盘确认），组 ID 去末两段。教训：改分组逻辑
  必须先看真实数据，不能按想象中的格式写
- **零方差组预览（前 3 步 48 组）：reward_std_mean 0.248、
  零方差组 47.9%**——近半 rollout 无学习信号，run2 全程实测后
  run3 评估开 filter_groups（动态采样）
- 步时 1218s（128 条轨迹），125 步全程序约 42h
- 结尾 DataLoader worker killed 的 Traceback 是退出期析构噪音（weakref），无害

