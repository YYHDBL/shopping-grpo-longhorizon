# V2 修复交接

## 版本边界

- V2 是当前活动版本，只保留 `Baseline → SFT → GRPO → Evaluation` 工作流程。
- V1 保留在 Git 历史或者参考分支，用于核对行为和结果。V2 完成替代后，清理活动目录中的 V1 启动脚本、测试、数据和说明文档。
- 建议使用独立的 V2 worktree 隔离开发，同时复用已经验证的 V2 数据处理和环境实现。worktree 负责隔离文件，仍需逐项修复下列代码问题。
- 旧版 Final-200 退出 V2。V2 正式评测集建立后，必须与训练数据按稳定任务标识严格去重。
- 修复期间禁止启动训练、合并模型和运行正式评测。

## 已验证状态

- `outputs/sft_dataset/train.parquet` 包含 5,852 条数据；`arguments` 和 `properties` 已使用 Arrow map，工具参数没有补入 `None`。
- `teacher` 列已经写入，GLM 为 2,991 条，DeepSeek 为 2,861 条。
- `ShoppingMultiTurnSFTDataset` 从新路径直接实例化成功，首条样本能够生成 token 和 loss mask。
- token 数量统计已经完成：中位数 4,860，百分之九十五分位数 13,767，最大值 42,455；超过 16,384 的数据有 196 条，超过 32,768 的数据有 9 条。
- `tests/test_verl_adapter.py` 已有 18 项通过记录。`test_smoke_shop_env.py` 的 3 项和 `test_verl_dynamic_sampling_patch.py` 的 4 项可以单独通过。
- V1 LoRA SFT 文件已经恢复，相关 24 项测试可以通过；这些文件仍被旧版入口引用，不属于 V2 最终结构。

## 阻塞修复

### 1. 固定可运行的依赖版本

当前 `.venv` 使用 `torch 2.11.0+cu130`、`vllm 0.25.1`、`verl 0.8.0` 和 `transformers 5.5.1`，驱动 535 无法运行这组 CUDA 构建。V2 依赖版本固定为：

- `torch==2.9.0`
- `vllm==0.12.0`
- `verl==0.8.0`
- `transformers>=4.56,<5`

依赖更换范围仅限项目 `.venv`，不得修改系统环境、驱动或者其他用户环境。安装后执行版本检查、依赖一致性检查和导入检查。`vllm 0.12.0` 还需验证 `tool_choice=required`、Qwen3.5 工具调用解析和 agent loop 行为。

### 2. 修复 veRL 自定义数据集入口

`scripts/train_sft.sh` 当前设置：

```text
data.custom_cls.path=shopping_grpo.training.sft.multiturn_dataset
```

veRL 0.8.0 会把没有 `pkg://` 的值解释为文件路径，当前配置会产生 `FileNotFoundError`。改用 veRL 支持的包路径：

```text
pkg://shopping_grpo.training.sft.multiturn_dataset
```

必须使用 veRL 自身的 `load_extern_object` 验证入口，不能只通过 Python import 验证。

`scripts/export_sft_parquet.py` 仍使用 veRL 原生 `MultiTurnSFTDataset` 验证数据。这里应调用训练入口实际使用的 `ShoppingMultiTurnSFTDataset`，确保导出验证和训练读取经过同一条代码路径。

### 3. 完成训练格式约束

- `max_length=16384` 配合 `truncation=error` 时，当前 196 条超长数据会使训练终止。构建数据时应按最终 chat template 的 token 数量删除这些完整轨迹，生成 5,656 条训练数据，并核对实际数量和删除比例。禁止截断工具调用轨迹。
- `properties` 的 map value 当前只保存 `type`，`finish_without_purchase.reason` 的 `enum=["no_suitable_product"]` 已经丢失。Parquet 必须保存完整工具 schema，并验证渲染后的工具定义与 `SHOP_TOOL_SCHEMAS` 一致。
- `ShoppingMultiTurnSFTDataset` 尚未读取 `enable_thinking` 并传给 `apply_chat_template`。所有渲染位置必须使用相同的模板参数。测试应明确 Qwen3.5 在禁用 thinking 时加入的边界 token，并验证 loss mask 完整覆盖 assistant 生成区间。
- 两遍渲染需要同时断言完整 assistant 前缀和生成提示前缀一致，防止模板变化造成 token 边界偏移。
- 为新数据集增加一个 CPU 测试，覆盖 map 读回归一化、无空参数、finish reason schema、assistant loss mask、超长过滤和 veRL `custom_cls` 加载。

### 4. 明确 SFT 训练配置

`scripts/train_sft.sh` 仍会继承部分 veRL 默认值，其中包括训练批量、学习率和模型精度。V2 训练脚本必须明确写出双卡 FSDP、BF16、训练批量、微批量、梯度累积、学习率、最大长度、训练轮数、保存频率和检查点选择规则，避免默认配置随依赖版本变化。

当前没有验证数据入口，需要在训练配置中明确开发集评测和检查点选择方式。脚本中的 `$PYTHONPATH` 应改为兼容未设置环境变量的写法 `${PYTHONPATH:-}`。

### 5. 消除全量测试的模块名冲突

`src/shopping_grpo/acceptance/hard_checks.py` 把 ShopSimulator 目录插入 `sys.path[0]`，导致其中的 `scripts` 包遮蔽仓库根目录的 `scripts` 包。单独测试可以通过，全量收集会产生 `scripts.*` 导入错误。

删除全局 `sys.path` 修改，通过正常包导入复用 ShopSimulator 代码。仓库根目录脚本只保留命令入口，共享实现放在 `shopping_grpo` 包内。修复后运行全部 CPU 测试和完整测试收集。

### 6. 更新 V2 数据契约

- 已知有 63 个 SFT 任务与旧版 `data/evaluation/tasks.jsonl` 重叠。旧版评测退出 V2 后，这些记录只用于迁移核对；新的 V2 评测集仍须满足严格去重要求。
- 当前 SFT 数据包含 5,391 条 `gold_purchase`、455 条 `partial_alternative` 和 6 条 `valid_alternative`，全部为 `reward_valid=true`。建议在 V2 契约中明确：正式评测的严格成功仍要求完整 `gold_purchase`；SFT 可以接收经过 JEV 审核的 alternative 轨迹，并保留审核依据和完整终止结果。
- 修改 `AGENTS.md` 和 V2 设计文档，使训练数据准入、正式评测准入和 Reward v3 终止条件使用相同定义。活动目录不再保留已经退出的旧版评测说明。

### 7. 清理 V1 活动入口

先根据调用关系确认 V2 已覆盖所需能力，再从 V2 活动目录删除 V1 LoRA SFT 入口、废弃测试和旧版数据引用。`scripts/train_lora_sft.py`、`src/shopping_grpo/collection/sft.py` 和 `src/shopping_grpo/smoke.py` 当前仍引用 V1 实现，需要一并处理，避免维持两套活动工作流程。

`tests/test_shopping_reward.py` 测试的是废弃的 Reward V1 行为，删除状态可以保留。`src/shopping_grpo/training/grpo/adapter/runtime.py` 中 `visited_subpages` 的缩进需要整理。

## 完成条件

- veRL 通过真实 `custom_cls` 配置加载数据集，并能够读取全部保留数据。
- 训练 Parquet 不含超过 16,384 token 的轨迹，工具 schema、thinking 参数和 loss mask 通过 CPU 测试。
- 全量测试能够完成收集，全部 CPU 测试通过。
- `.venv` 依赖检查通过；vLLM 冒烟测试覆盖工具强制调用、工具解析和 agent loop。
- V2 活动目录只有一套训练入口和一套 Reward v3 契约。
- 训练、模型合并和正式评测保持未执行状态。
