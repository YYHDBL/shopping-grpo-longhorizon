# 实验 01：Qwen3.5-2B Instruct 的零样本 Shopping Agent 基线

> 状态：首轮完成，原始输出待从服务器同步入库｜日期：2026-07-20｜训练仓库 commit：`c741c03`

## 问题

在不给任何本项目购物轨迹做 SFT 的情况下，`Qwen/Qwen3.5-2B` 这个具备对话与 tool calling 能力的 Instruct 模型，能否直接完成 ShopSimulator 的单轮购物任务？

这里的“Base”是实验语境：**尚未进行购物 SFT 的 Instruct 基线**，不是 `Qwen3.5-2B-Base` 裸预训练权重。

## 先排除工程假象

这次评测前完成了两项独立验证：

1. ShopSimulator 单环境连续 3 次 `reset → release_one` 都能回收同一个 env slot；
2. 用本项目 12 个 tool schema、`tool_choice="required"` 与 vLLM `qwen3_coder` parser 发出最小请求，模型返回 1 个 `search_products` tool call，生成仅 26 token。

因此，后续的失败不能简单归因于“模型不支持函数调用”。早期的 Hermes parser 配置错误和未设置输出上限造成过 HTTP 400 与长输出卡死；本次使用 `qwen3_coder`，并由客户端固定 `max_tokens=512`。

## 评测协议

| 项目 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-2B` Instruct |
| 任务 | `data/benchmarks/shop_benchmark_v1.jsonl`，固定 200 条 held-out task |
| 任务隔离 | 与当前 380 条 SFT task 无重叠 |
| 环境 | ShopSimulator HTTP 服务，单 env slot |
| 推理 | vLLM OpenAI-compatible API，`qwen3_coder` parser |
| 解码 | temperature=0、top_p=1、max_tokens=512 |
| Agent 上限 | max_steps=35，每 task 一次 rollout |
| 成功定义 | 环境完成购买，且 `r_type/r_att/r_option/r_price` 全部为 1 |

执行分为 10 条 smoke 和完整 200 条。smoke 没有 hang，环境正常回收；完整运行约 35 分钟并正常退出。期间出现一次单 slot 不可用，执行 `release_all` 后续跑成功。这个事件应保留在记录中，但不能被解释为模型成功率。

## 观察到的结果

| 指标 | 结果 |
|---|---:|
| 已记录 task | 200 / 200 |
| 严格成功率 | 0.0% |
| 环境 done rate | 0.5%（1 / 200） |
| 平均终局 reward | 0.0025 |
| 平均工具步数 | 6.7 |

| 终局状态 | 数量 | 占比 |
|---|---:|---:|
| `invalid_action_limit` | 159 | 79.5% |
| `error` | 29 | 14.5% |
| `max_steps` | 10 | 5.0% |
| `assistant_final` | 1 | 0.5% |
| `done` | 1 | 0.5% |

动作守卫共拦截了 711 次。其中 `click_not_in_previous_observation` 有 582 次，`search_not_available_on_current_page` 有 128 次，`select_option_is_navigation_button` 有 1 次。

## 这说明了什么，没说明什么

**已经能说明：** 这个 2B Instruct 模型能生成合法的工具调用格式，但没有掌握本环境最基本的 grounded action 策略：它经常根据旧页面或臆测的按钮继续点击，而不是只从最新 observation 中选择可用目标。三次连续被守卫拒绝后终止，正是这种错误的集中表现。零样本下 0% 严格成功率是 SFT 冷启动必要性的直接证据。

**还不能说明：** 29 条 `error` 是否全部是模型决策错误。它们可能混有 vLLM 返回、解析或服务端网络问题；正式论文式对比前，必须从 `raw.jsonl` 对这 29 条按 error type 再分组。无论如何，即便只看 159 条明确的 guard 终止，也足以说明零样本策略不可用。

## 下一步决定

1. 不修改这份 benchmark 的任务、prompt、guard 或成功标准；它是后续 SFT/GRPO 的固定对照。
2. 将服务器的 `raw.jsonl`、`summary.json`、vLLM 启动命令和 ShopSimulator 日志路径同步到对应实验输出目录，以便第三方复查本记录。
3. 以当前 353 条可在 24K 模板下训练的 SFT 样本启动 LoRA SFT；训练后用同一 200 条任务重跑，重点比较 `click_not_in_previous_observation` 和严格成功率是否下降/上升。

这不是一个“Base 模型不好”的泛泛结论，而是一个可检验的假设：如果 SFT 有效，模型首先应该学会尊重最新页面 observation，再谈更难的商品属性与规格判断。
