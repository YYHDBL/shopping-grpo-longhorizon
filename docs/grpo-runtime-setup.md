# Vanilla GRPO 服务器执行手册（veRL 0.8）

这份文档用于在 GPU 服务器上建立一套**全新的 GRPO 环境**并完成 1 个更新步的 smoke。不要继续修补 `.venv-sft`、`.venv-infer` 或半成品 `.venv-grpo`；它们曾被 reference veRL fork 的 editable install 和手工 Transformers patch 污染。

## 1. 固定版本与取舍

| 组件 | 固定版本 | 原因 |
|---|---:|---|
| Python | 3.12 | 与服务器镜像和 vLLM wheel 对齐 |
| veRL | 0.8.0 | 已有 AgentLoop、Qwen3.5 actor patch 和内置 `qwen3_coder` parser |
| vLLM | 0.25.1 | Qwen3.5 需要新版 vLLM |
| PyTorch | 2.11.0 | vLLM 0.25.1 的正式 wheel 固定依赖 |
| Transformers | 5.11.0 | 支持 Qwen3.5，同时避开 veRL 后续已标记的不兼容版本 |
| Ray | 2.56.1 | 满足 veRL 0.8，固定当前稳定版本 |
| TensorDict | 0.10.0 | veRL 0.8 支持范围的最高版本 |
| NumPy | 2.2.6 | vLLM 0.25.1 的 OpenCV 依赖要求 NumPy 2 |

不要安装 `verl[vllm]`。veRL 0.8 的这个 extra 仍声明旧的 `vllm<=0.12.0`，会把 Qwen3.5 所需的新 vLLM 降级。项目改为安装 `verl==0.8.0` 核心包和独立固定的 `vllm==0.25.1`。

veRL 0.8 的包元数据还声明 `numpy<2`，但上游源码已更新为 `numpy>=2`。安装时使用项目内的 override 文件覆盖这条过期约束；不要手工拆包或修改 site-packages。

## 2. 拉取本项目并新建干净环境

以下命令都在数据盘执行，不修改三个旧 venv：

```bash
cd /root/autodl-tmp/shopping-grpo-longhorizon
git fetch origin
git switch agent/vanilla-grpo-runtime
git pull --ff-only origin agent/vanilla-grpo-runtime

export UV_CACHE_DIR=/root/autodl-tmp/.cache/uv
export HF_HOME=/root/autodl-tmp/.cache/huggingface
unset PYTHONPATH

uv venv --python 3.12 .venv-grpo-v080
uv pip install \
  --python .venv-grpo-v080/bin/python \
  --torch-backend=auto \
  -r requirements-grpo.txt \
  --override requirements-grpo-overrides.txt

source .venv-grpo-v080/bin/activate
```

禁止执行下面几类操作：

- 不要 `uv pip install -e .../agentic-grpo-longhorizon/verl`；
- 不要把 reference fork 写入 `PYTHONPATH`；
- 不要手工修改 `transformers/__init__.py`；
- 不要在 `.venv-sft` 或 `.venv-infer` 中继续安装 GRPO 依赖。

## 3. 验证导入来源和版本

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import verl

for name in ("verl", "vllm", "torch", "transformers", "ray", "tensordict", "numpy"):
    print(f"{name}={version(name)}")
print("verl_source=", Path(verl.__file__).resolve())
PY
```

期望：

```text
verl=0.8.0
vllm=0.25.1
torch=2.11.0
transformers=5.11.0
ray=2.56.1
tensordict=0.10.0
numpy=2.2.6
```

`verl_source` 必须位于 `.venv-grpo-v080/.../site-packages/verl`，不得出现 `agentic-grpo-longhorizon`。

项目配置显式关闭 `use_remove_padding`。当前 Qwen3.5 + SDPA 路线不依赖另行编译的 FlashAttention 2，避免重复 SFT 阶段的 CUDA 编译问题。

项目还启用了 `model.lora.merge=true`：训练参数仍然只有 LoRA，但 rollout 前临时把 LoRA 合入基座再同步标准权重。这样可避开 Qwen3.5 在 vLLM 原生 LoRA 权重同步中的 `base_layer` 命名不兼容。

veRL 0.8 在 old-log-prob 前处理阶段仍会无条件导入 `flash_attn.bert_padding`，即使 `use_remove_padding=false`。项目通过 Ray worker setup hook 复用 veRL 已内置的纯 PyTorch padding 实现；这只替换索引和 padding 工具，模型 attention 仍使用 SDPA，不需要安装 FlashAttention。

## 4. 启动并验证 ShopSimulator

ShopSimulator 继续使用它自己的 Python 3.10 环境。GRPO 的默认批次是 `2 prompt × 4 rollout`，所以必须初始化至少 8 个环境槽：

```bash
cd /root/autodl-tmp/ShopSimulator/shop_env/shop_env

# 先激活服务器上已经验证通过的 ShopSimulator venv，再执行：
python -c \
  'import pack_api; pack_api.env_max_num=8; pack_api.initialize_environments(); pack_api.app.run(host="127.0.0.1", port=5700)'
```

回到训练仓库，用一次 reset/release 验证结构化接口和租约回收：

```bash
cd /root/autodl-tmp/shopping-grpo-longhorizon
source .venv-grpo-v080/bin/activate
export PYTHONPATH="$PWD/src"
export SHOPSIM_BASE_URL=http://127.0.0.1:5700

python - <<'PY'
import os
from shopping_grpo.shop_http_env import ShopAgentEnv

env = ShopAgentEnv(base_url=os.environ["SHOPSIM_BASE_URL"], timeout=60)
try:
    result = env.reset(0)
    print("reset_ok=", bool(result.get("instruction") or result.get("observation")))
finally:
    env.release()
print("release_ok=True")
PY
```

## 5. 准备模型和 parquet

`GRPO_MODEL_PATH` 必须指向已经合并 SFT LoRA 的完整 checkpoint，不是 adapter 目录。若 train/val parquet 尚未生成，按 [运行手册](runbook.md) 中的命令生成。正式 benchmark 不得用作 GRPO validation。

```bash
export GRPO_MODEL_PATH=/root/autodl-tmp/checkpoints/qwen35-2b-shopping-sft-v2-merged
export GRPO_TRAIN_FILE=/root/autodl-tmp/shopping-grpo-longhorizon/data/verl/grpo_train_v1.parquet
export GRPO_VAL_FILE=/root/autodl-tmp/shopping-grpo-longhorizon/data/verl/grpo_val_v1.parquet
export GRPO_OUTPUT_DIR=/root/autodl-tmp/checkpoints/qwen35-2b-shopping-grpo-smoke
export SHOPSIM_BASE_URL=http://127.0.0.1:5700
```

先只运行预检。它会在加载模型权重前检查 parquet、Python、依赖版本、CUDA、veRL 来源、内置 parser 和项目 AgentLoop：

```bash
PYTHONPATH=src python scripts/check_grpo_runtime.py
```

## 6. 运行 1 步 smoke

```bash
bash scripts/run_vanilla_grpo.sh \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1
```

本次 smoke 的验收条件：

1. 实际生成 2 组 prompt，每组 4 条 rollout；
2. 模型能输出 `qwen3_coder` tool call，ShopSimulator 能返回 observation；
3. 只有环境正常 `done && over` 时使用原生终局 reward，其余终止 reward 为 0；
4. 正常、异常、超步数和模型提前结束后，8 个环境租约都被释放；
5. veRL 完成 1 次 policy update 后正常退出，无 HTTP 400、环境槽耗尽或旧 fork 导入。

若失败，保留完整 traceback、上面的版本与 `verl_source` 输出，不要现场手改 site-packages。

## 7. 为什么不再使用 `verl.interactions`

veRL 0.8 已没有 reference fork 的 `verl.interactions`。同时，0.8 默认 `BaseTool` 会在**每次工具调用**后执行 release，而 ShopSimulator 要求一整条 trajectory 独占同一环境。

本项目因此只做一层必要适配：

```text
ShoppingToolAgentLoop.run
  → reset(task_id)，绑定 trajectory-local env/state
  → 复用 veRL 0.8 ToolAgentLoop 和内置 qwen3_coder parser
  → 多步执行项目统一 tools
  → 写入 ShopSimulator 原生终局 reward
  → finally release
```

reference 仓库今后只用于理解实验流程，不再进入运行时依赖。

## 8. 版本依据

- [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)：Qwen3.5 需要新版 vLLM，并使用 `qwen3_coder` 工具协议；
- [veRL v0.8.0 setup.py](https://github.com/verl-project/verl/blob/v0.8.0/setup.py)：确认核心依赖范围，以及旧 `verl[vllm]` extra 的版本上限；
- [veRL v0.8.0 ToolAgentLoop](https://github.com/verl-project/verl/blob/v0.8.0/verl/experimental/agent_loop/tool_agent_loop.py)：确认 AgentLoop 状态机、工具生命周期和 reward 输出接口；
- [veRL v0.8.0 tool_parser.py](https://github.com/verl-project/verl/blob/v0.8.0/verl/experimental/agent_loop/tool_parser.py)：确认 `qwen3_coder` 已由 veRL 内置。
