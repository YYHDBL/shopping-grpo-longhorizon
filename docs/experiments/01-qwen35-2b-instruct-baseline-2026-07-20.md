# 实验 01：Qwen3.5-2B Instruct 零样本 Shopping Agent 基线

> 状态：已完成 + 深度分析｜日期：2026-07-20｜训练仓库 commit：`c04caf1`

## 问题

在不给任何本项目购物轨迹做 SFT 的情况下，`Qwen/Qwen3.5-2B` Instruct 模型能否直接完成 ShopSimulator 的单轮购物任务？

这里的"Base"是实验语境：**尚未进行购物 SFT 的 Instruct 基线**，不是裸预训练权重。

## 先排除工程假象

本次评测前已完成两项独立验证：

1. ShopSimulator 单环境连续 3 次 `reset → release_one` 都能回收同一个 env slot
2. 用本项目 12 个 tool schema、`tool_choice="required"` 与 vLLM `qwen3_coder` parser 发出最小请求，模型返回 1 个 `search_products` tool call，仅 26 token

因此，后续的失败不能简单归因于"模型不支持函数调用"。早期的 Hermes parser 配置错误（XML 格式 tool call）和未设置输出上限已修复——本次使用 `qwen3_coder`，客户端 `max_tokens=512`。

## 评测协议

| 项目 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-2B` Instruct |
| 任务 | `data/benchmarks/shop_benchmark_v1.jsonl`，固定 200 条 held-out task |
| 任务隔离 | 与当前 380 条 SFT task 无重叠（random seed 20260720） |
| 环境 | ShopSimulator HTTP 服务，单 env slot (env_max_num=1) |
| 推理 | vLLM OpenAI-compatible API，`qwen3_coder` parser |
| 解码 | temperature=0、top_p=1、max_tokens=512 |
| Agent 上限 | max_steps=35，每 task 一次 rollout |
| 成功定义 | 环境完成购买，且 r_type/r_att/r_option/r_price 全部为 1 |
| 输出目录 | `outputs/eval/qwen35_2b_instruct_v2/` |

执行分两阶段：10 条 smoke test（验证无 hang、环境回收正常）→ 完整 200 条（约 35 分钟）。中途出现一次单 slot 不可用（`env_max_num=1` 未释放），执行 `release_all` 后续跑成功。

## 整体结果

| 指标 | 值 |
|---|---:|
| 已记录 task | 200 / 200 |
| 严格成功率 | **0.0%** |
| 环境 done rate | 0.5%（1 / 200） |
| 平均终局 reward | 0.0025 |
| 平均工具步数 | 6.7 |
| 总动作守卫拦截次数 | **711** |

| 终局状态 | 数量 | 占比 |
|---|---:|---:|
| `invalid_action_limit` | 159 | 79.5% |
| `error` | 29 | 14.5% |
| `max_steps` | 10 | 5.0% |
| `assistant_final` | 1 | 0.5% |
| `done` | 1 | 0.5% |

## 深度失败分析

### 1. Error 类型细分（29 条）

| Error 类型 | 数量 | 说明 |
|---|---|---|
| **HTTPError (400)** | **27** | vLLM 返回 Bad Request |
| ShopEnvironmentError | 1 | 环境槽位不可用（已修复） |
| KeyError | 1 | 解析异常 |

27/29 的 error 是 vLLM HTTP 400。进一步检查发现这些轨迹在 error 发生前都成功执行了多步 tool call（平均 15.7 步），最后一步 assistant message 也包含合法的 tool_calls 结构（不含 XML 格式）。这意味着 HTTP 400 不是早期的 parser 格式问题，更可能是：
- 长对话上下文超出 max-model-len 触发了服务端截断或格式错误
- vLLM qwen3_coder parser 在特定参数组合下产生的序列化格式不兼容

这 27 条不能简单归为"模型不懂工具调用"——它们在报错前已经成功执行了多轮交互。

### 2. 动作守卫拦截——失败发生在极早期

拦截的 179 条轨迹中，**首次被拒绝的步骤分布**：

| 步骤 | 轨迹数 | 累计占比 |
|---|---:|---:|
| step 1 | 44 | 24.6% |
| step 2 | 47 | 50.8% |
| step 3 | 74 | 92.2% |
| step 4-11 | 14 | 100% |

**92.2% 的失败轨迹在前 3 步就被首次拦截。** 这意味着问题不在长链推理，而在最基本的"读页面 → 选正确按钮"环节。

`invalid_action_limit` 轨迹的平均有效步数仅为 **3.3 步**（max=13），说明模型在极少步数后就把连续 3 次守卫宽容度用完。

### 3. 三种典型失败模式

#### 模式 A：打开商品后立刻搜索（57 条，占拦截的 31.8%）

```
step 0: search_products("...") ✓     → 搜索结果页
step 1: open_product("B0XXXXXX") ✓  → 商品详情页
step 1: search_products("...") ✗     → "搜索功能是否可用: False"
step 2: open_product("invalid") ✗    → 继续点击错误按钮
step 2: open_product("invalid") ✗    → 三次拦截，终止
```

模型打开商品后无视当前页面是否支持搜索，直接再次搜索。这说明模型**不会根据 observation 动态调整可用工具集**。

#### 模式 B：在无子页按钮的页面点击 Features/Attributes（51 条）

```
step 0: search_products ✓
step 1: open_product ✓
step 1: view_features ✗ → 商品详情页没有 Features 按钮
```

模型"猜测"每个商品都有 Features/Attributes 子页，而不是先确认按钮存在再调用。

#### 模式 C：无限翻页不打开商品（约 10 条 max_steps）

```
step 0: search_products ✓
step 1-34: next_page × 14 次，翻到最后一页也不打开任何商品
```

模型在搜索结果页一直翻页，从不打开商品查看。全部 35 步耗尽。这是最极端的不理解任务目标的表现。

### 4. 唯一 "done" 轨迹（task 21713）

```
step 0: search_products → 0  reward
step 1: open_product   → 0  reward
step 2: view_description → 0 reward
step 3: prev_page       → 0  reward
step 4: select_option   → 0  reward
step 5: buy_now         → 0.5 reward → done=true

reward_detail: {r_type: 1.0, r_att: 0.5, r_option: 0.0, r_price: True}
category_match: False ← 买了错误品类
```

虽然成功走到了 `buy_now` 并让环境结束，但**商品类别根本不对**（`category_match: False`）。这说明模型能执行购买流程，但不能将用户需求与商品正确匹配。

## 核心结论

| 能做的 | 不能做的 |
|---|---|
| 生成合法 tool call JSON | **根据 observation 判断当前页面有哪些可用操作** |
| 执行 search → open → select → buy 流程 | 打开商品后抑制再次搜索的冲动 |
| 在简单 case 中完成购买（1/200） | 区分搜索结果页和商品详情页的工具可用性 |
| — | 将用户需求中的约束与商品实际属性进行匹配 |

这不是一个"Base 模型不好"的泛泛结论。具体来说：**79.5% 的失败是因为模型在打开商品后立即尝试不可用的操作**（搜索或点击不存在的 Features 按钮），然后连续重复犯错被守卫终止。Qwen3.5-2B Instruct 知道如何输出 tool call JSON 格式，但**完全不知道购物场景下的工具使用规则**。

27 条 HTTP 400 错误的位置提示，即便模型在正确工具调用序列中走到了 ~15 步，仍然会因上下文增长导致服务端错误——这是下一步 vLLM 配置需要关注的问题（可能需增大 max-model-len 或调整 parser 行为），但不应与模型策略能力混淆。

## SFT 需要解决的具体问题（可检验假设）

如果 SFT 有效，模型应首先在以下几点出现统计显著改善：

1. **搜索后打开商品** → 不再在商品详情页重新搜索（减少 `search_not_available_on_current_page` 从 128 → ?）
2. **打开商品后只使用当前页面的按钮** → 不再盲点 Features/Attributes（减少 `click_not_in_previous_observation` 从 582 → ?）
3. **翻页后打开商品** → 不再无限翻页（减少 `max_steps` 从 10 → ?）
4. **购买前核验类别** → 不再买错品类（`r_type` 从 0.5% → ?）

## 下一步

1. 用当前 353 条 SFT 样本启动 LoRA SFT（见实验 02）
2. 训练后用同一 200 条 benchmark 重跑，重点对比上述四个具体指标
3. 在 SFT 完成后检查 27 条 HTTP 400 是否消失——如果消失，说明是模型 token 使用效率导致的上下文超限；如果仍在，需要排查 vLLM 侧问题
