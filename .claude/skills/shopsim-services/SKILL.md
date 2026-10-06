---
name: shopsim-services
description: 购物 Agent V2.0 项目的服务地图：GPU 卡位、训练容器、ShopSimulator 环境服务、模型权重与 API 端点的位置和起停方法。接手本项目的 agent 先读这里。
---

# ShopSim V2.0 服务地图

仓库 `/data/jyh-yyh/shopping-grpo-longhorizon`（分支 `V2.0`）。所有 key 在仓库根 `.env`（git-ignored），不要写进任何提交文件。

## 1. GPU 卡位（8×A800-80GB，宿主机编号）

| 卡 | 归属 | 状态 |
|---|---|---|
| 0-1 | 用户自己的 `gemma-4-31B-it` 容器 | **禁止触碰** |
| 2-3 | 用户自己的 `JYHLLM_30B` | **禁止触碰** |
| 4-7 | 本项目训练用（容器 `shopping-gpu`） | 训练/空闲 |

红线：绝不停/删别人的容器（`gemma-4-31B-it` 等），绝不用 `docker stop` 清理不属于本项目的容器。占用卡 4-7 前先 `nvidia-smi` 确认为空。

## 2. 训练容器 `shopping-gpu`

宿主机驱动 535 跑不了 cu130，一切训练在容器内（镜像自带 CUDA 兼容层）：

```bash
# 已在跑：docker ps | grep shopping-gpu。若需重建：
docker run -d --name shopping-gpu --entrypoint sleep \
  --gpus '"device=4,5,6,7"' -v /data/jyh-yyh:/data/jyh-yyh d5219758abb3 infinity
# 进入
docker exec -it shopping-gpu bash
```

容器内栈：verl 0.9.1 + vllm 0.25.0 + torch 2.11+cu130 + transformers 5.10.4（Python 3.12，已装好，别重装）。
**坑**：当前容器 NVIDIA_VISIBLE_DEVICES=all，能看到全部 8 张物理卡（容器内 nvidia-smi 的 0-7 就是物理 0-7，别误把 0-3 的占用当成自己的）。选卡一律用 UUID（launch 脚本里有现成的物理 4-7 UUID 列表）；容器内 nvidia-smi 显示的 PID 是宿主机 PID，容器 /proc 里查不到属正常。
**坑**：宿主机 `.venv` 不认 Qwen3.5，数据处理用 `.venv/bin/python`，训练/推理一律容器内。

## 3. ShopSimulator 环境服务（pack_api）

训练 rollout 必须的环境服务。Flask，唯一路由 `POST /api/shop_agent`，动作：`reset` / `interact` / `release_one` / `release_all`。`/health` 之类不存在，404 不代表挂了。

两个实例（网络隔离，互不干扰）：
- **容器内 160 槽 @5700**（训练用。**槽位数必须 ≥ train_batch_size × n**：run1 8×8=64 需要 80；run2 16×8=128 需要 160。并发超槽会产生抢槽失败的基础设施轨迹）：
  ```bash
  docker exec -d shopping-gpu bash -c "cd /data/jyh-yyh/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/shop_env && SHOPSIM_ENV_SLOTS=160 SHOPSIM_PORT=5700 python3 pack_api.py > /tmp/pack_api_160.log 2>&1"
  ```
  注意用 `docker exec -d`（`bash -c "... &"` 内嵌后台会被 exec 退出连带杀掉）
- 宿主机 5700（tmux `shopsim-env`，采集期遗留 16 槽，一般不用管）

验证（真实调用才算数）：
```bash
docker exec shopping-gpu curl -s -X POST http://127.0.0.1:5700/api/shop_agent \
  -H 'Content-Type: application/json' -d '{"action":"reset","idx":0}'
# 返回 env_idx + instruction 即正常；测完 release_one 释放槽位
```
80 槽初始化要几分钟（日志 "Environment N is being initialized" 走到 79 才监听）。

**⚠ kill 训练/评测后必做 `release_all`**（第 14 号坑）：被 kill 的 rollout 的 session 租约
不会自动释放，槽位池被死租约占满 → 下一个任务（评测/训练）报
`Unable to get available environment resource`。实测：kill run4 训练后直接起评测，
8 题即挂（256 槽被训练遗留租约占满）。起新任务前先：
```bash
docker exec shopping-gpu curl -s -X POST http://127.0.0.1:5700/api/shop_agent \
  -H 'Content-Type: application/json' -d '{"action":"release_all"}'
```

## 4. 模型权重与训练数据

| 路径 | 大小 | 用途 |
|---|---|---|
| `models/Qwen3.5-9B/` | 19G | Base 原始权重（基线评测/回滚） |
| `checkpoints/sft_v2_run1/global_step_172/` | 53G | SFT 最终步（veRL 格式） |
| `outputs/models/sft-merged/` | 18G | SFT 合并后 HF 格式（**GRPO 起点**、SFT 评测用） |

**训练数据（唯一合法来源）**：
- GRPO：`outputs/grpo/train_1000_curriculum.parquet`（1000 题，run3 用）或 `train_1000.parquet`（run1/run2 用）；验证 `outputs/grpo/smoke.parquet`（64 题）
- **`data/grpo/*.parquet` 已弃用禁止使用**：实测与冻结评测集重叠 43 题、与 SFT 教材重叠 249 题（隔离契约违规）。重排脚本 `scripts/reorder_grpo_curriculum.py` 含隔离自校验可参考

## 5. 模型 API 端点（key 全在 `.env`）

| 端点 | 用途 |
|---|---|
| OpenRouter `/api/alpha/decisions`，模型 `typesafe/jev-1.13` | Jev 判定（数据验收/评测/reward 灰区，标准 v3 已冻结） |
| `TEACHER_BASE_URL`（bigmodel）GLM-5.3-Flash | 双 Teacher 之一（采集已完成） |
| `TEACHER2_BASE_URL`（opencode）DeepSeek V4.1 Flash | 双 Teacher 之二 |
| SwanLab（`SWANLAB_API_KEY`） | 训练监控云端面板 |

三个端点均服务器直连，客户端代码已绕过代理（ProxyHandler({})），不依赖 SSH 隧道。

## 6. 训练启动

- **GRPO 正式跑**：容器内 tmux 会话执行对应 launch 脚本（`outputs/launch_grpo_run1.sh` / `launch_grpo_run2.sh`，UUID 选卡 4-7，SwanLab 云端）
- **任意时刻停训保存**：`touch /data/jyh-yyh/shopping-grpo-longhorizon/outputs/SAVE_AND_STOP`——trainer 在下一个安全点（步入口或优势计算后，rollout 不改权重）保存 checkpoint 并退出，最多损失当前半步 rollout。等 5~25 分钟看到日志出现 SAVE_AND_STOP detected 即完成
- SFT 已完成（172 步，val loss 0.292→0.278），不要重跑
- 启动前检查：环境服务活（见 §3）、卡 4-7 空、`df /data` 有空间（每个 GRPO checkpoint ~53G）

**tmux 纪律**：长任务一律 tmux 里跑；退出 tmux 用 `Ctrl+B` 再 `D`（直接关终端/Ctrl+C 会杀前台训练进程）；进容器跑训练用 `docker exec -it shopping-gpu bash` 后再起 tmux 或直接跑。

## 7. 历史坑（报错先对照这里）

1. NCCL "Duplicate GPU detected" → 已修：compat hook 里 `PYTORCH_NVML_BASED_CUDA_CHECK=1`（勿删 `src/shopping_grpo/training/grpo/compat.py` 的这行）
2. vLLM wake OOM → gpu_memory_utilization 0.35（0.5 必炸）
3. 训练 forward OOM → ppo_mini_batch_size 8（16 炸）、optimizer_offload 必开
4. 序列截断断言 → ppo_max_token_len 24576（实测合法轨迹 18k+）
5. DataLoader worker 死 → num_workers=0
6. flash-attn 没装 → sdpa（`override_config.attn_implementation: sdpa`）
7. `torch.cuda.is_available()` 会假 True → 验证 GPU 必须真做矩阵乘
8. 长序列训练前向 OOM（log_softmax 的 [seq,vocab] logits 峰值）→ `use_fused_kernels: true` + `fused_kernel_options.impl_backend: torch`（512 token 分块输出头，峰值 56.8→43.8G）；生效标志是初始化日志 `Using Torch backend for fused kernels`
9. 评测起 vLLM 服务必须带 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`，否则第一通 tool_choice 请求 400
10. veRL 0.9.1 `algorithm.disable_kl` 是死字段（源码无读取点），ref 加载实际由 `use_kl_loss`/`use_kl_in_reward` 控制
11. colocate 崩溃后清显存：要连 `VLLM::EngineCore` 一起杀（不止 VLLM::Worker）
12. GRPO checkpoint 目录比 SFT 多一层 `actor/`，合并时 `--local_dir` 指到 `global_step_N/actor`
13. **容器僵尸堆积会杀死 NVML**：容器 PID 1 是 sleep 不收尸，训练崩溃/停止后僵尸越积越多（实测 470→1300+），NVML 初始化枚举进程时直接 "Failed to initialize NVML: Unknown Error"（宿主机正常）。解法：`docker restart shopping-gpu`（清零），然后重启环境服务（160 槽）+ ckpt-prune tmux。长训练后起任何新 GPU 进程前先 `nvidia-smi -L` 验一下

## 8. 项目暂停状态（2026-10-06）

**项目已暂停，卡 4-7 已归还用户自己的服务。**

- `gemma-4-31B-ita`（31B，TP4，端口 18003，`NVIDIA_VISIBLE_DEVICES=4,5,6,7`）已恢复运行
- 我们的容器 `shopping-gpu` 已 **stop（未删除）**，随时可恢复：

```bash
# 恢复项目：起容器 → 起环境服务 → 起训练（checkpoint 随时可续）
docker start shopping-gpu
docker exec -d shopping-gpu bash -c "cd /data/jyh-yyh/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/shop_env && SHOPSIM_ENV_SLOTS=256 SHOPSIM_PORT=5700 python3 pack_api.py > /tmp/pack_api_256.log 2>&1"
# 训练：outputs/launch_grpo_run4.sh（resume auto，从 checkpoints/grpo_run4 最新存档续）
# 或评测：outputs/launch_grpo_run3* 系列脚本参考
```

- **卡 0-3 的服务一直未被触碰**（gemma-it / JYHLLM 正常）
- 恢复训练前：确认用户已同意占用卡 4-7（需临时停 gemma-4-31B-ita）
