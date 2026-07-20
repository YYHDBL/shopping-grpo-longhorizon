# 已验收 SFT 数据快照

此目录发布完整的正式采集批次，便于复现实验：`raw.jsonl.gz`、`accepted.jsonl`、`rejected.jsonl`、`reject_stats.json` 与可直接训练的 `sft.jsonl`。不包含 API 密钥、模型 checkpoint 或环境服务数据。

- 快照版本：2026-07-20
- 样本数：380 条 accepted trajectory
- 格式：`sft.jsonl` 每行一条标准 OpenAI messages JSON，包含 `trajectory_id`、`task_id`、`messages` 与 `tools`
- 验收规则：工具调用合法、执行购买、环境结束，且 `r_type`、`r_att`、`r_option`、`r_price` 均为 1；这些隐藏验收字段不会写入训练 messages。
- SHA-256：`7b1d82ed8c18c6fbd23af89a574258d040c4c6db8d1ba9d9512fc1e418abc3fa`

未压缩的 `raw.jsonl` 为 118MB，超过 GitHub 普通 Git 的单文件限制；仓库发布其 gzip 版本 `raw.jsonl.gz`（约 14MB）。需要原始 JSONL 时执行 `gzip -dk raw.jsonl.gz`。未压缩源只保留在本地，便于断点续跑。

这是当前阶段的可复现训练快照，不是最终 500 条目标。后续数据扩充会以新快照和新校验和发布，避免静默改写已训练的数据版本。
