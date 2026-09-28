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
