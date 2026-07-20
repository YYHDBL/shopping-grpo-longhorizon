# 公开数据说明

`shop_tasks.jsonl` 是从 ShopSimulator 导出的 23,421 个公开 task_id 清单，用于确定性选择采集任务；每行只有一个 `task_id`，不含用户画像、奖励或商品库隐藏答案。

- SHA-256：`35a5978bba829a6b4196f0e4ce65d5f7d366cc0704469e3b7f8a6d02f50a9dea`

可训练的已验收轨迹快照位于 [../outputs/flash_accepted_500_parallel/sft.jsonl](../outputs/flash_accepted_500_parallel/sft.jsonl)，其格式和数据隔离规则见同目录 README。
