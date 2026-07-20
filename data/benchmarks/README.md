# ShopSimulator benchmark v1

`shop_benchmark_v1.jsonl` 固定包含 200 个单轮 ShopSimulator `task_id`，用于公平比较 Base、SFT 和未来 GRPO 模型。

- 随机种子：`20260720`
- SFT 排除集：当前 `outputs/flash_accepted_500_parallel/sft.jsonl` 中的 380 个 task
- 推理协议：temperature=0、max_steps=35、每 task 一次 rollout
- 主指标：严格成功率。只有环境完成购买且 `r_type`、`r_att`、`r_option`、`r_price` 均为 1 才计成功；未运行 task 同样计入分母。
- 辅助指标：环境完成率、四项 reward 分量通过率、平均步数、动作守卫拒绝原因。
- 清单 SHA-256：`9905ab9f4b8d9bbbc44adfd8cc4de2bce2797a63366dc94950015a5eed86655b`

不要把该清单中的 task 用于后续 SFT 或 GRPO 训练采集。若需要新增 benchmark，创建新的版本文件，不能静默改写 v1。
