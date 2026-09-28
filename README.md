# Shopping Agent V2.0

面向长程购物 Agent 的可复现后训练与评测项目。

```text
数据验收 → 画像与切分 → 双 Teacher 教材采集 → veRL 全参 SFT → 在线 GRPO → 冻结评测
```

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![veRL](https://img.shields.io/badge/veRL-0.9.1-0E8A16)](https://github.com/verl-project/verl)
[![vLLM](https://img.shields.io/badge/vLLM-0.25.0-0E6DB8)](https://github.com/vllm-project/vllm)
[![ShopSimulator](https://img.shields.io/badge/Environment-ShopSimulator%20v2.1-4C78A8)](https://arxiv.org/pdf/2601.18225)

## 环境与依赖

- Python 3.12 容器路线，依赖由 `pyproject.toml` / `uv.lock` 固定（uv 管理）
- `verl==0.9.1`、`vllm==0.25.0`、`torch==2.11.0`、`transformers>=5.5.3,<5.11,!=5.6.0`
- 模型：`models/Qwen3.5-9B/`（后训练版，全参 BF16 双卡 FSDP）
- API Key（Teacher / Jev）在 `.env`，参考 `.env.example`

```bash
bash scripts/setup.sh        # 安装环境依赖与 ShopSimulator
```

## 工作流程

| 阶段 | 说明 | 入口 |
|---|---|---|
| Baseline | 未训练底座在冻结 eval 切分上的基线轨迹 | `bash scripts/serve_model.sh models/Qwen3.5-9B` + `python scripts/evaluate_model.py --name baseline` |
| 数据验收 | 两级验收（本地硬约束 + Jev 语义）+ 复查定案 | `scripts/run_acceptance.py` |
| 画像与切分 | 三条件画像分配、冻结 eval 切分 | `scripts/build_split.py` |
| Teacher 采集 | 双 Teacher 分治，每题预算 3 次成功即停 | `scripts/collect_teacher_data.py` |
| 教材加工 | Student 版提示词移植 + 逐消息 train 标记 + 真实 token 超长过滤 | `scripts/build_sft_dataset.py` + `scripts/export_sft_parquet.py` |
| SFT 训练 | veRL fsdp_sft + 自定义多轮工具数据集 | `bash scripts/train_sft.sh` |
| GRPO | 在线 RL，Reward v3 终局信号 | `scripts/train_grpo.py` |
| Evaluation | 训练后检查点的冻结评测（严格成功指标） | `bash scripts/serve_model.sh <checkpoint>` + `python scripts/evaluate_model.py --name <run>` |

评测入口统一使用 `scripts/evaluate_model.py`：环境由 `scripts/start_environment.sh`
启动，模型经 `scripts/serve_model.sh` 以 OpenAI 兼容接口 serve；轨迹落盘
`outputs/evaluation/<name>/trajectories.jsonl`（支持断点续跑），严格成功等指标
汇总在 `summary.json`。正式评测需用户明确指令后才执行。

设计决策与执行记录见 `docs/plans/`（基线、交接、数据验收记录、工作总结）。

## 评测契约

- 正式评测严格成功 = 完整 `gold_purchase` 终止结果且 `reward_valid=true`。
- SFT 教材准入 = `reward_valid=true` 且（gold 或经冻结 Jev 判定 `fully_satisfies` 的替代购买）。
- 训练数据与冻结 `tag=eval` 评测集严格去重；旧版 Final-200 已退出 V2，仅存于 Git 历史作迁移核对。

训练、模型合并与正式评测需用户明确指令后才执行。
