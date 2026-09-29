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
- **容器内 80 槽 @5700**（训练用，rollout 并发 64 需要它）：
  ```bash
  docker exec shopping-gpu bash -c "cd /data/jyh-yyh/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/shop_env && SHOPSIM_ENV_SLOTS=80 SHOPSIM_PORT=5700 nohup python3 pack_api.py > /tmp/pack_api_container.log 2>&1 &"
  ```
- 宿主机 5700（tmux `shopsim-env`，采集期遗留 16 槽，一般不用管）

验证（真实调用才算数）：
```bash
docker exec shopping-gpu curl -s -X POST http://127.0.0.1:5700/api/shop_agent \
  -H 'Content-Type: application/json' -d '{"action":"reset","idx":0}'
# 返回 env_idx + instruction 即正常；测完 release_one 释放槽位
```
80 槽初始化要几分钟（日志 "Environment N is being initialized" 走到 79 才监听）。

## 4. 模型权重

| 路径 | 大小 | 用途 |
|---|---|---|
| `models/Qwen3.5-9B/` | 19G | Base 原始权重（基线评测/回滚） |
| `checkpoints/sft_v2_run1/global_step_172/` | 53G | SFT 最终步（veRL 格式） |
| `outputs/models/sft-merged/` | 18G | SFT 合并后 HF 格式（**GRPO 起点**、SFT 评测用） |

## 5. 模型 API 端点（key 全在 `.env`）

| 端点 | 用途 |
|---|---|
| OpenRouter `/api/alpha/decisions`，模型 `typesafe/jev-1.13` | Jev 判定（数据验收/评测/reward 灰区，标准 v3 已冻结） |
| `TEACHER_BASE_URL`（bigmodel）GLM-5.3-Flash | 双 Teacher 之一（采集已完成） |
| `TEACHER2_BASE_URL`（opencode）DeepSeek V4.1 Flash | 双 Teacher 之二 |
| SwanLab（`SWANLAB_API_KEY`） | 训练监控云端面板 |

三个端点均服务器直连，客户端代码已绕过代理（ProxyHandler({})），不依赖 SSH 隧道。

## 6. 训练启动

- **GRPO 正式跑**：容器内 tmux 会话执行 `bash /data/jyh-yyh/shopping-grpo-longhorizon/outputs/launch_grpo_run1.sh`（125 步 ~17h，UUID 选卡 4-7，SwanLab 云端）
- SFT 已完成（172 步，val loss 0.292→0.278），不要重跑
- 启动前检查：环境服务活（见 §3）、卡 4-7 空、`df /data` 有空间（每个 GRPO checkpoint ~18G，save_freq=25 共 5 个）

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
