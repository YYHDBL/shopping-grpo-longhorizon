# V2 轨迹样例

每个 JSONL 文件随机抽取 100 条完整轨迹；每行是一条轨迹对象。

| 文件 | 来源 | 抽样范围 | 抽样种子 |
|---|---|---|---:|
| `teacher_attempts_random_100.jsonl` | `outputs/collection/trajectories/` | Teacher 全部单次尝试轨迹文件 | 20261019 |
| `grpo_run3_train_random_100.jsonl` | `outputs/rollout_data/run3/` | GRPO run3 的 step 1–50 训练 rollout | 20261020 |
| `grpo_run3_step50_eval_random_100.jsonl` | `outputs/evaluation/grpo-run3-step50/trajectories.jsonl` | 1,092 条冻结评测轨迹 | 20261021 |

Teacher 样本按单次尝试抽取，同一任务可能出现多次尝试。训练样本只从确认属于 run3 的 step 1–50 抽取。评测样本公开后会暴露部分冻结任务及其轨迹，不应再把这 100 道样本当作未公开的盲测题。
