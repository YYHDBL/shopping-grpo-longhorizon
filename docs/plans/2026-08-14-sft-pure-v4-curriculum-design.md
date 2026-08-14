# SFT Pure V4 累积式课程训练设计

## 目标

将 `data/sft_pure_v4/all.jsonl` 作为唯一 SFT 主数据源，在不启动训练的前提下，
准备可在服务器上一条命令执行的质量审计、固定划分和三阶段课程训练。

## 质量合同

全量 1,192 条轨迹必须通过确定性硬门槛：

- `task_id` 唯一，并且与 `data/evaluation/tasks.jsonl` 零重叠。
- 轨迹与难度标签 task ID 一一对应。
- 每个 assistant 回合最多一个工具调用，参数是合法 JSON object。
- 调用的工具名存在于该行的 tool schema。
- 无 Guard rejection 文本，无终局 Reward / Gold 文本泄漏。
- 最后一个 assistant 动作是且仅有一次 `buy_now`，并有正常终局 tool 回复。

长轨迹、搜索较多、重复动作，以及“标签说需要比较但只打开一个候选”
不作自动删除，而是记入 review flags。这些可能是有效难例，必须有人工或更强
证据才能判定为坏轨迹。

## 划分

使用固定 seed `20260814`，先根据技能将任务放入三个原子 bucket，再在每个
bucket 内稳定抽取 10% 作为开发集。开发集只用于 loss 和小规模轨迹评测，
永远不参与 SFT 梯度更新。

| Bucket | 定义 | 预期全量 | 训练 / 开发 |
|---|---|---:|---:|
| foundation | `difficulty=simple` | 284 | 256 / 28 |
| constraints | medium，不需要 query rewrite 或候选比较 | 603 | 543 / 60 |
| strategy | 其余 medium 和全部 hard | 305 | 274 / 31 |
| 合计 |  | 1,192 | 1,073 / 119 |

Final-200 Clean 不用于划分、调参或 checkpoint 选择，只用于最终横向比较。

## 课程

| Stage | 累积数据 | 预期训练数 | Epoch | Learning rate | 主要能力 |
|---|---|---:|---:|---:|---|
| A | foundation | 256 | 1 | `1e-4` | 合法调用、打开、选规格、及时购买 |
| B | foundation + constraints | 799 | 1 | `7e-5` | 品类、预算、规格轴、variant 价核对 |
| C | 全部训练数据 | 1,073 | 1 | `5e-5` | 搜索改写、候选比较、防循环、长轨迹收口 |

每阶段训练一个新 LoRA adapter，完成后合并到当前基座，下一阶段以该 merged
checkpoint 继续。这使基础题分别出现三次、硬条件题出现两次、策略难题出现
一次，同时逐阶段降低学习率。

Stage C merged checkpoint 是 GRPO 的唯一 SFT 起点。Stage A/B 只是可审计的课程中间产物。

## 服务器执行合同

```bash
bash scripts/sft_curriculum.sh
```

该命令依次完成 A/B/C 训练和 adapter merge。支持 `--dry-run`、从某阶段继续、
停在某阶段以及 SwanLab。默认不删除中间 checkpoint，避免自动清理导致数据丢失。

## 选 checkpoint

Validation loss 只用于发现训练异常。最终选择应使用独立开发任务的：

- 工具合法率和非终局继续动作率。
- 规格、价格硬门槛成功率。
- 严格 `gold_purchase` 成功率。
- `repeat_loop` / `max_steps` / 无工具 final 的比例。
- foundation / constraints / strategy 三个 bucket 的分项成功率。
