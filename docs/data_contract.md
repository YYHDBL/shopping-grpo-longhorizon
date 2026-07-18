# 数据约定

`collect_teacher_rollouts.py` 会为每个 `(task_id, attempt_index)` 追加一条 raw JSON。raw 以唯一的 `trajectory_id` 标识，并保留 `messages`、已执行的 `steps`、`initial_result`、`terminal_result`、`status`、`done`、`final_reward` 和错误信息，便于追溯。

`build_sft_data.py` 从完整 raw JSONL 确定性生成四类文件：

- `accepted.jsonl`：通过全部规则的原始轨迹。
- `rejected.jsonl`：只保存 `trajectory_id`、`task_id`、`status` 和结构化 `reject_reasons`。
- `stats.json`：总数、通过数、拒绝数和拒绝原因计数。
- `sft_openai_messages.jsonl`：仅保存 `trajectory_id`、`task_id`、清洗后的 `messages` 和共用 `tools`。

一条轨迹必须无错误、执行购买、得到环境终局结果，且 `reward_detail` 的 `r_type`、`r_att`、`r_option`、`r_price` 均为 1，才能被接受。此外，每个工具调用必须是 JSON 对象、不得含 tool schema 未声明字段、与映射后的环境动作一致，并且来自紧邻的 observation。

为避免“先选规格、再换商品”污染训练数据，首次 `select_option` 后只允许继续 `select_option`（补选规格）或 `buy_now`。这条规则只用于验收和 SFT 筛选；rollout 运行时仍由当前环境 observation 决定模型可执行的动作。

SFT 行会排除 `goal`、标准答案、`purchase`、`reward_detail`、终局 reward 等环境隐藏字段。教师推理会转写为 assistant `content` 中的 `<think>` 标签，而非保留 provider 专有字段。数据可直接传给 Hugging Face `apply_chat_template(..., tools=row["tools"])`。
