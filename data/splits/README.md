# GRPO task split v1

这里的清单不是离线 RL 轨迹数据。它们只保存 `task_id`，实际训练时由当前 policy 在线进入 ShopSimulator rollout。

- `grpo_probe_pool_v1.jsonl`：2,000 个候选题。它与已发布的 Teacher raw rollout（757 个 task）及 `shop_benchmark_v2_50` 完全不重叠。
- `grpo_train_v1.jsonl`：尚未生成。先用冻结 SFT policy 跑完候选池，再按实际执行工具步数精确抽取 1,000 个：short 300（≤10 步）、medium 450（11–20 步）、long 250（≥21 步）。基础设施错误不进桶；任何桶不足时脚本会失败，不会偷偷用其他长度补齐。

如果在已发布 raw snapshot 之外又新增并**冻结**了一批 SFT 数据，生成候选池时额外传入 `--exclude-sft path/to/sft.jsonl`。不要把仍在采集中的本地输出写进正式 manifest。
