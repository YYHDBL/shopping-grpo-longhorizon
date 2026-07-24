# SFT v3 短轨迹补充批次

该目录保存用于下一轮 LoRA SFT 的新增数据快照。

- 采集模型：DeepSeek V4 Flash，未开启 thinking
- 原始轨迹：317 条
- 严格验收成功：100 条
- 拒绝：217 条
- 成功轨迹长度：4～15 个环境步骤，平均 9.08 步
- 验收要求：完成购买，且 `r_type`、`r_att`、`r_option`、`r_price` 均为 1
- SFT 格式：Action-only；已移除 guard 拒绝消息、错误工具调用、Teacher reasoning 和终局隐藏反馈
- 工具 schema：禁止未声明参数，并强化当前页面动作约束

文件说明：

- `raw.jsonl.gz`：完整原始轨迹的 gzip 快照；解压后为 `raw.jsonl`
- `accepted.jsonl`：100 条完整成功轨迹
- `rejected.jsonl`：未通过轨迹及结构化拒绝原因
- `reject_stats.json`：验收统计
- `sft.jsonl`：下一轮 SFT 直接使用的数据

SHA-256：

- `raw.jsonl.gz`：`a39fb2738d62ef2f5f80adcae1aa2312615fce320f7cafe4b1e692dcd136adb0`
- `accepted.jsonl`：`83fc35e61a546050785ccb7fe95cdaaed8eecc768a3a3997e4b2074e23ab5359`
- `sft.jsonl`：`2cc5f11ec8e5526ac07503275f4175aadfc674d7f19b21e48f203150131ea21b`

未压缩的 `raw.jsonl` 仅保留在本地用于断点续跑，避免 Git 仓库无谓膨胀。
