# 实验 05：Qwen3.5 + veRL 0.8 + ShopSimulator 的 GRPO 运行时踩坑实录

> 状态：96GB 单卡 one-step smoke 已闭环；8 条 rollout、Reference 和 optimizer update 均完成｜日期：2026-07-23

## 摘要

这次工作的目标看似很小：先在单张 48GB GPU 上定位问题，再在 RTX PRO 6000
Blackwell 96GB 上完成一个 Vanilla GRPO smoke。批次固定为
`2 prompt × 4 rollout = 8 trajectory`，只执行一次 Qwen3.5-2B LoRA optimizer
update。

实际困难不在 GRPO 公式，而在多个框架边界同时发生变化：

- veRL 的 reference fork 与官方 0.8 API 不同；
- Qwen3.5 需要新版 Transformers 和 vLLM；
- veRL 0.8 的已发布 NumPy 元数据已经落后于新版 vLLM；
- ShopSimulator 要求一整条 trajectory 独占环境，不能按单次 tool call 释放；
- FSDP actor 与 vLLM rollout 之间还要同步 Qwen3.5 LoRA 权重；
- 一些依赖直到 rollout 完成后的 old-log-prob 阶段才会被延迟导入，普通 import
  preflight 无法提前发现。

最终结果是：

1. 全新、可复现的 veRL 0.8 运行环境建立成功；
2. 官方内置 `qwen3_coder` parser、Ray、vLLM、CUDA 和项目 AgentLoop 均通过预检；
3. `model.lora.merge=true` 修复了 Qwen3.5 原生 LoRA 权重同步的
   `base_layer` 命名问题；
4. 实际生成了 8 条 ShopSimulator 轨迹，8 个环境全部释放，无 HTTP 400、槽位耗尽或 OOM；
5. `bdec522` 在 rollout 后计算 `old_log_prob` 时，veRL 进入
   `left_right_2_no_padding → unpad_input`，因缺少 `flash_attn` 失败；
6. `679e09c` 用 Ray worker setup hook 接入 veRL 自带的纯 PyTorch padding fallback，
   绕过了不必要的 `flash_attn` 依赖；
7. 随后的实际瓶颈不是 Reference，而是 actor 为 8 条长轨迹重算 `old_log_prob` 时，
   为 entropy 指标物化了约 32GB 的完整词表临时张量；
8. `a14538f` 开启 vLLM rollout log-prob，并使用 veRL 0.8 官方
   `rollout_correction.bypass_mode`，跳过 actor 重算；
9. 96GB one-step smoke 最终完成 `training/global_step=1` 和 optimizer update，正常
   exit 0，8 个环境全部释放。

这篇记录重点说明哪些做法有效、哪些“看起来能修”的做法反而会扩大变量，以及下一次
smoke 应把门禁放在哪里。

---

## 1. 任务和框架边界

### 1.1 训练闭环

本项目的在线训练链路是：

```text
veRL RayPPOTrainer
  → FSDP LoRA actor
  → vLLM async rollout
  → qwen3_coder tool parser
  → ShoppingToolAgentLoop
  → ShopSimulator HTTP environment
  → native terminal reward
  → GRPO advantage / old log prob / policy update
```

这里不是普通的单轮文本 GRPO。每个 sample 都是一条多步购物轨迹，模型要反复读取
observation、生成工具调用并推进同一个有状态环境。

### 1.2 固定运行时

最终固定矩阵如下：

| 组件 | 版本 |
|---|---:|
| Python | 3.12.3 |
| veRL | 0.8.0 |
| vLLM | 0.25.1 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.11.0 |
| Ray | 2.56.1 |
| TensorDict | 0.10.0 |
| NumPy | 2.2.6 |
| ShopSimulator Python | 3.10.20 |

训练环境固定为 `.venv-grpo-v080`。ShopSimulator 使用自己的 Python 3.10 venv，
两套依赖完全隔离。

### 1.3 最小 smoke 约束

```text
train_batch_size = 2
rollout.n = 4
max_num_seqs = 8
agent.num_workers = 8
ShopSimulator env_max_num = 8
total_training_steps = 1
validation = off
checkpoint save = off
```

这组约束非常重要。smoke 的目标是验证闭环，不是顺手开始完整训练。遇到失败后扩大
batch、增加 step 或直接跑 500-step，只会让故障证据更模糊。

---

## 2. 坑一：旧虚拟环境“能 import”不代表能训练

服务器上已有 `.venv-sft`、`.venv-infer` 和 `.venv-grpo`。这些环境曾经混入：

- reference veRL fork 的 editable install；
- 手工修改过的 Transformers；
- 指向 reference fork 的 `PYTHONPATH`；
- SFT、推理和 GRPO 不同阶段安装的包。

最危险的现象不是立刻 import error，而是同一进程中不同模块来自不同目录。例如主程序
可能来自 reference fork，Ray worker 却从旧推理 venv 加载 torch 或 vLLM。这样的环境
有时能跑到模型加载，之后才在远程 worker 中报一个看似无关的错误。

### 修复

不再修补旧环境，直接新建：

```bash
unset PYTHONPATH
uv venv --python 3.12 .venv-grpo-v080
uv pip install \
  --python .venv-grpo-v080/bin/python \
  --torch-backend=auto \
  -r requirements-grpo.txt \
  --override requirements-grpo-overrides.txt
```

然后把 `verl.__file__` 作为硬门禁：

```text
/root/autodl-tmp/shopping-grpo-longhorizon/
  .venv-grpo-v080/lib/python3.12/site-packages/verl/__init__.py
```

只检查 `verl.__version__` 不够。reference fork 可能复用相同或不可比较的版本号，实际
文件路径才能证明运行时来源。

### 经验

涉及 Ray 时，环境污染会被放大，因为 driver、actor、rollout server 和 worker 都会
重新 import。应在启动 Ray 前就拒绝错误来源，不要等远程 traceback 才猜包从哪里来。

---

## 3. 坑二：reference fork 的流程不能直接翻译成官方 veRL 0.8 API

早期参考代码使用 `verl.interactions` 和项目自带的 Qwen3-Coder parser。官方 veRL
0.8 已经发生两个关键变化：

1. `verl.interactions` 不再存在；
2. `qwen3_coder` 已经内置在
   `verl.experimental.agent_loop.tool_parser.ToolParser`。

如果继续把 reference fork 放入 `PYTHONPATH`，表面上可以找回旧 API，实际会让官方
0.8 的 trainer、AgentLoop 和 worker 与旧 fork 类型混在一起。这不是兼容层，而是两套
运行时叠加。

### 修复：只保留最小项目适配层

项目删除重复 parser，直接使用 veRL 内置 `qwen3_coder`。自定义代码只负责
ShopSimulator 特有的生命周期和 reward：

```text
ShoppingToolAgentLoop.run
  → reset(task_id)，绑定 trajectory-local env/state
  → 复用 veRL 0.8 ToolAgentLoop 状态机
  → 执行项目统一工具和动作守卫
  → 记录 ShopSimulator 原生终局 reward
  → finally release
```

preflight 同时验证：

- `ShoppingToolAgentLoop` 是官方 `ToolAgentLoop` 子类；
- `ShopSimulatorTool` 是官方 `BaseTool` 子类；
- `AgentState.TERMINATED` 和关键 lifecycle API 存在；
- `qwen3_coder` 在官方 parser registry 中。

### 经验

迁移框架版本时，不要用 `PYTHONPATH` 把消失的 API“补回来”。应先判断该能力是被删除、
改名还是已经上移到官方实现，再把项目适配层缩到最小。

---

## 4. 坑三：ShopSimulator 的租约单位是 trajectory，不是 tool call

veRL 0.8 的默认 `BaseTool` 生命周期会在每次工具调用后执行 release。但
ShopSimulator 的一条购物轨迹必须从 reset 到终止始终占用同一个环境，否则下一次
工具调用会落到不同页面状态。

### 修复

- `ShopSimulatorTool.release()` 不释放真实 HTTP 环境；
- 真实租约由 `ShoppingToolAgentLoop.run()` 管理；
- `finally` 中统一 release；
- 正常完成、assistant 提前结束、动作守卫连续拒绝、工具异常和 max steps 都走同一
  清理路径。

同时 ShopSimulator 必须在独立 Python 3.10 环境中启动至少 8 个槽位：

```bash
python -c \
  'import pack_api; pack_api.env_max_num=8; pack_api.initialize_environments(); \
   pack_api.app.run(host="127.0.0.1", port=5700)'
```

### 验证方法

训练前不能只 reset 一次。实际做了三层验证：

1. 连续 16 次 `reset → release`，确认槽位按 `0..7` 循环复用；
2. 同时 reset 8 个 client，确认获得 8 个唯一 env index；
3. 全部 release 后再次同时租用 8 个槽位，确认没有泄漏。

### 经验

有状态 agent 环境的核心接口不是 `execute(tool)`，而是“谁拥有会话、所有权持续多久、
所有退出分支如何释放”。在接框架之前先画清租约生命周期，比先写 tool schema 更重要。

---

## 5. 坑四：固定版本本身也可能构成不可解的依赖集合

最初 `requirements-grpo.txt` 固定了：

```text
vllm==0.25.1
numpy==1.26.4
```

安装器给出的证据是：

```text
opencv-python-headless>=4.13.0.90 depends on numpy>=2
vllm>=0.25.1 depends on opencv-python-headless>=4.13.0
```

仅删除 `numpy==1.26.4` 仍然无效，因为 veRL 0.8 的已发布包元数据还声明
`numpy<2`。但 veRL 后续官方源码已经改为 `numpy>=2`，说明这里是旧元数据而不是本项目
代码依赖 NumPy 1.x。

### 修复

仓库同时固定：

```text
# requirements-grpo.txt
numpy==2.2.6

# requirements-grpo-overrides.txt
numpy==2.2.6
```

并使用 uv 的显式 override：

```bash
uv pip install \
  --python .venv-grpo-v080/bin/python \
  --torch-backend=auto \
  -r requirements-grpo.txt \
  --override requirements-grpo-overrides.txt
```

`uv pip check` 会依据 veRL 0.8 的旧 wheel metadata 报告 NumPy 冲突，因此不再作为硬
门禁。取而代之的是实际 import、CUDA、parser、AgentLoop 和 smoke。

### 经验

override 不是“忽略所有冲突”。它只适合已经有上游证据、范围单一且运行时可验证的旧
元数据。override 文件必须进仓库，不能只存在于某台服务器的 shell history。

---

## 6. 坑五：模型、adapter、训练数据和 benchmark 的边界容易混淆

GRPO 初始模型必须是已经合并 SFT LoRA 的完整 checkpoint：

```text
checkpoints/qwen35-2b-shopping-sft-v2-merged/
  config.json
  model.safetensors
  tokenizer.json
  ...
```

不能把 SFT adapter 目录直接当 `GRPO_MODEL_PATH`。GRPO 会以 merged SFT 模型为基座，
再挂载一枚新的 GRPO LoRA，从而保留 SFT 和 RL 两阶段的权重边界。

正式 GRPO train split 还要求先用冻结的 SFT policy probe 2,000 个候选任务，再按实际
工具步数分层冻结 1,000 个 task。最终 benchmark 不能拿来做 validation。

本次 smoke 不扩大成 2,000-task probe，只使用候选池中的 2 个 train task：

```text
23321
11346
```

另取两个不重叠 task 生成仅用于满足 preflight 的 val parquet，且
`trainer.val_before_train=false`。这两个临时 parquet 不是正式 GRPO 数据资产。

### 经验

“只跑一个 step”并不意味着可以随便从 benchmark 抽两条。即使是 smoke，也要保持
train、validation、benchmark 和 adapter/base checkpoint 的语义边界。

---

## 7. 坑六：Qwen3.5 原生 LoRA 权重同步中的 `base_layer` 命名不兼容

第一轮基于 commit `d5bfa52` 的 smoke 已经做到：

- 预检通过；
- FSDP actor 加载成功；
- vLLM server 启动成功；
- AgentLoop workers 创建成功。

但在 actor 首次向 vLLM 同步 LoRA/FSDP 权重时失败：

```text
update_weights_from_ipc
  → receiver.receive_weights
  → model.load_weights
  → Qwen3_5ForConditionalGeneration.load_weights
  → QKVParallelLinear.get_submodule(...)

AttributeError:
QKVParallelLinear has no attribute `base_layer`
```

这个错误发生在 ShopSimulator reset 之前，所以该轮实际 trajectory 数为 0。继续修改
tool parser、HTTP 协议或环境槽位都不会触及根因。

### 修复

commit `bdec522` 启用 veRL 官方配置：

```yaml
actor_rollout_ref:
  model:
    lora_rank: 16
    lora_alpha: 32
    target_modules: all-linear
    lora:
      merge: true
```

这里的含义不是改成 full-parameter training。actor 仍然只训练 rank-16 LoRA；在 rollout
同步前，veRL 临时将 LoRA 合入基座并向 vLLM 发送标准权重，从而绕开 vLLM 原生 LoRA
loader 对 `base_layer` 名称的假设。

### 验证结果

`bdec522` 后：

- Hydra 实际展开配置显示 `lora.merge=True`；
- actor 仍是 `PeftModelForCausalLM`；
- `base_layer` 错误没有复现；
- vLLM 正常进入生成；
- 2 个 prompt 各 reset 4 次，共产生 8 条轨迹。

### 经验

权重同步错误要先判断发生在“训练模型结构”“序列化名称”还是“推理 loader”。如果
actor 已成功加载，而错误只在 vLLM `load_weights` 出现，盲目修改环境或重新合并 SFT
checkpoint 都是错误方向。

---

## 8. 坑七：preflight 全绿，仍可能在 rollout 之后遇到延迟依赖

`bdec522` smoke 的 8 条轨迹全部完成并 release：

| task_id | rollout 数 |
|---:|---:|
| 11346 | 4 |
| 23321 | 4 |

8 个 env index `0..7` 每个使用一次，每条轨迹有 31–35 次环境交互。没有 HTTP 400、
环境槽耗尽、OOM 或无限生成。峰值显存为 30,928 MiB。

但 trainer 在 rollout 后计算旧策略 log probability 时失败：

```text
RayPPOTrainer._compute_old_log_prob
  → left_right_2_no_padding
  → unpad_input
  → _get_attention_functions
  → from flash_attn.bert_padding import ...

ModuleNotFoundError: No module named 'flash_attn'
```

项目配置已经设置 `use_remove_padding: false`，文档路线也选择 Qwen3.5 + SDPA、不另行
编译 FlashAttention 2。但 veRL 0.8 的这条 old-log-prob 前处理路径仍然进入了
`left_right_2_no_padding`，并延迟 import `flash_attn`。

### 为什么 preflight 没发现

当前 preflight 验证的是：

- 包版本和文件来源；
- CUDA 可用；
- vLLM/Ray/Transformers/veRL 可 import；
- AgentLoop 类型和 parser registry；
- parquet 文件存在。

它没有构造 rollout batch，也没有执行 `_compute_old_log_prob`。`flash_attn` 又是在该
函数深处延迟 import，因此所有静态 import 检查都可以通过。

### 最小修复

commit `679e09c` 没有安装 FlashAttention，也没有改动 veRL site-packages，而是在每个
Ray worker 启动时安装一个窄范围兼容 hook：

```yaml
ray_kwargs:
  ray_init:
    runtime_env:
      worker_process_setup_hook: shopping_grpo.verl_compat.install_torch_padding_fallback
```

hook 只把 `attention_utils._get_attention_functions()` 指向 veRL 已经内置在
`npu_flash_attn_utils` 中的纯 PyTorch `index_first_axis`、`pad_input`、
`rearrange` 和 `unpad_input`。模型 attention 仍走 SDPA；这里替换的只是 padding 与
索引辅助函数。

preflight 也会显式安装一次 fallback，单元测试则验证返回的四个函数确实来自 veRL
内置实现。这样既避免编译新的 CUDA 扩展，也没有扩大依赖矩阵。

这次仍然没有采取以下现场操作：

- 没有直接 `pip install flash-attn`；
- 没有重新安装一套 CUDA；
- 没有修改 veRL site-packages；
- 没有为了“先跑起来”切换未知版本；
- 没有把 `use_remove_padding` 改成与文档相反的值碰运气。

后续实测证明这个 fallback 能覆盖 Ray worker 的延迟 import，但它并不能解决长轨迹
old-log-prob 重算本身的显存峰值。这个区别很重要：`flash_attn` import error 消失只说明
执行路径继续向前，不代表该路径适合当前 batch。

---

## 9. 坑八：真正的 OOM 在 actor old-log-prob 重算，不在 Reference

开启 padding fallback 后，48GB 和 96GB 的 smoke 都能生成并释放 8 条轨迹，随后在：

```text
RayPPOTrainer._compute_old_log_prob
  → DataParallelPPOActor.compute_log_prob
  → entropy_from_logits
```

发生 OOM。96GB 机器当时已经接近满显存，entropy 还要申请约 32.06GiB。原因是配置
`rollout.calculate_log_probs=false`，trainer 只能让 actor 对 8 条长轨迹重新前向；
即便 `entropy_coeff=0`，veRL 0.8 仍会为统计指标计算完整词表 entropy。

### 为什么关闭 Reference 不是正确的第一步

OOM 发生在 `_compute_old_log_prob`，Reference log-prob 尚未开始。LoRA 配置下 Reference
还是同一个 actor 临时关闭 adapter，并非另一份常驻完整模型。因此先关闭 KL/Reference
既不能消除当前 32GB 临时张量，也会改变原实验设计。

### 最小修复：复用 rollout policy 的 log-prob

veRL 0.8 已有官方 bypass 路径，可以直接把 vLLM rollout 时记录的
`rollout_log_probs` 作为 PPO 的 `old_log_probs`：

```yaml
actor_rollout_ref:
  rollout:
    calculate_log_probs: true

algorithm:
  rollout_correction:
    bypass_mode: true
    rollout_is: null
    rollout_rs: null
    loss_type: ppo_clip
```

这保留 Reference/KL，同时完全跳过 `_compute_old_log_prob`。修复没有升级 veRL、没有修改
site-packages，也没有改模型或 batch。配置测试显式锁定这五个字段，防止以后回退到
actor 重算。

### 96GB one-step 实测

在 commit `a14538f` 的配置上，执行固定 one-step 命令后：

| 指标 | 实测值 |
|---|---:|
| prompt × rollout | `2 × 4 = 8` |
| task 11346 / 23321 | `4 / 4` |
| 环境释放 | `8 / 8` |
| `training/global_step` | `1` |
| `actor/ppo_kl` | `0.0059456001` |
| `update_actor` | `233.8509s` |
| 单步训练时间 | `290.7633s` |
| 端到端命令耗时 | `493s` |
| GPU 采样峰值 | `97,109MiB`（`94.83GiB`） |
| 进程退出码 | `0` |

日志中 `_compute_old_log_prob`、HTTP 400 和 CUDA OOM 的出现次数均为 0。训练结束后再次
同时租用 8 个 slot，得到唯一集合 `0..7`，随后全部 release；Ray、vLLM、5700 端口和
GPU 显存也全部清理。

本批两组 prompt 的 terminal reward 都是 0，因此 advantage、loss 和 grad norm 都是
有限值但为 0。optimizer/update_actor 路径和 `global_step=1` 已完成，证明运行时闭环
成立；但这批 smoke 没有提供非零学习信号，不能把它解读为“模型能力已改善”。这也是
为什么 smoke 的验收应区分“optimizer 确实执行”和“本批产生有效梯度”。

退出阶段还有一条不影响退出码的 PyTorch atexit 清理 traceback：
DataLoader worker 在 Ray teardown 中收到 `Killed`。主训练已先打印完整 step 指标和
`Final validation metrics: None`，命令 exit 0，且没有残留进程，因此它记录为清理噪声，
不是训练失败；如果正式长跑中再次出现，则应单独治理 worker teardown。

---

## 10. smoke 的故障推进

| Commit | 到达阶段 | 轨迹 | 环境释放 | 参数更新 | 结果 |
|---|---|---:|---:|---:|---|
| `d5bfa52` | actor→vLLM 首次权重同步 | 0 | 未租用 | 0 | `base_layer` 失败 |
| `bdec522` | rollout 后 old-log-prob | 8 | 8/8 | 0 | 缺少 `flash_attn` |
| `679e09c` | actor old-log-prob entropy | 8 | 8/8 | 0 | 48GB/96GB OOM |
| `a14538f` | optimizer + 正常退出 | 8 | 8/8 | 1 | one-step 闭环 |

这张表体现了一个重要原则：smoke 的价值不是只有“成功/失败”两个状态，而是每次修复后
闭环向前推进了多少。`bdec522` 没有完成 GRPO 更新，但它已经用实际 8 条 trajectory
证明了模型同步、parser、AgentLoop、HTTP 环境和 release 生命周期。

---

## 11. 哪些 preflight 值得保留

### 安装后

```python
from importlib.metadata import version
from pathlib import Path
import torch
import verl

for name in (
    "numpy", "verl", "vllm", "torch",
    "transformers", "ray", "tensordict",
):
    print(name, version(name))

print(Path(verl.__file__).resolve())
print(torch.cuda.is_available())
```

### 启动训练前

1. 确认 train/val parquet 都存在；
2. 确认 merged checkpoint 包含完整模型权重，不是 adapter；
3. 确认 `qwen3_coder` 在官方 parser registry；
4. 确认 8 个 ShopSimulator slots 可同时租用并全部回收；
5. 确认训练配置展开后仍是：

```text
train_batch_size=2
rollout.n=4
total_training_steps=1
val_before_train=false
save_freq=-1
test_freq=-1
lora.merge=true
rollout.calculate_log_probs=true
rollout_correction.bypass_mode=true
```

### 运行时取证

- 完整 stdout/stderr，不只保存最后 20 行；
- `pip freeze`；
- `nvidia-smi` 和 1 秒粒度显存曲线；
- 实际 `verl.__file__`；
- ShopSimulator reset/release 日志；
- 精确命令和 Git commit；
- 失败发生时的阶段：安装、preflight、环境、rollout、reward、old log prob 或 optimizer。

---

## 12. 这次明确证明“不该做”的事

1. 不在 `.venv-sft`、`.venv-infer` 或旧 `.venv-grpo` 上继续叠包。
2. 不安装 reference veRL fork 的 editable package。
3. 不把 reference fork 加入 `PYTHONPATH`。
4. 不手工修改 Transformers 或 veRL site-packages。
5. 不用 `verl[vllm]` extra；它的旧 vLLM 上限不适合 Qwen3.5。
6. 不把 SFT adapter 当成完整 GRPO base checkpoint。
7. 不用 benchmark 充当 GRPO validation。
8. 不在安装器给出明确冲突后盲目升级/降级整套依赖。
9. 不在一个 step 尚未闭环时扩大到完整 GRPO。
10. 不把“8 条 rollout 成功”写成“一次 GRPO 更新成功”。
11. 不在 OOM 尚未到达 Reference 阶段时先关闭 Reference。

---

## 13. 可迁移到其他 Agentic RL 项目的经验

### 12.1 先验证生命周期，再验证 reward

多步环境最容易坏在租约泄漏和异常退出。只要环境不能可靠回收，reward 设计得再漂亮也
无法稳定训练。

### 12.2 driver import 成功只是最浅的一层

Ray worker、vLLM engine、模型权重同步和 trainer 后处理都有自己的延迟 import。应按
执行阶段设计 preflight，而不是只写一段 `import torch, vllm, verl`。

### 12.3 框架适配要区分三类问题

```text
API 兼容：类、方法、状态机是否存在
权重兼容：actor 参数名能否被 rollout engine 接收
运行兼容：某条实际 batch 路径是否触发隐含 CUDA/FlashAttention 依赖
```

三类问题的 traceback 位置不同，修复手段也不同。把它们混成“veRL 版本不对”会导致
无休止的重装。

### 12.4 每次修复只移动一个变量

本次有效修复都遵循这一点：

- 旧环境污染 → 新建独立 venv；
- NumPy 元数据冲突 → 单一 override；
- reference parser → 官方内置 parser；
- tool-call release 不兼容 → AgentLoop 持有 trajectory lease；
- Qwen3.5 LoRA loader 不兼容 → 官方 `lora.merge=true`。

当错误到达新的阶段后立即停止，保存证据，再决定下一项单变量修复。

---

## 14. 当前结论

截至 commit `bdec522`，实际运行已经完成下面这段闭环：

```text
固定运行时
  → preflight
  → actor/FSDP 初始化
  → veRL→vLLM 标准权重同步
  → 2 prompt × 4 rollout
  → 8 条 ShopSimulator 多步轨迹
  → 8/8 release
```

随后由实际运行证明：

```text
vLLM rollout log-prob
  → bypass actor old-log-prob recompute
  → Reference log-prob
  → GRPO advantage
  → LoRA optimizer update
  → global_step=1
  → 正常 exit 0
```

因此当前准确状态是：**Vanilla GRPO one-step 运行时闭环已经通过。** 本批 reward 全 0，
所以这次只证明训练基础设施能够执行一次更新，不证明存在非零学习信号，也不应据此扩大
训练。正式实验前仍应在目标 96GB 卡上重复同一个 one-step smoke，并确认 reward 分布
符合预期。

完整服务器操作步骤见
[Vanilla GRPO 服务器执行手册](../grpo-runtime-setup.md)，任务冻结与数据边界见
[Vanilla GRPO v1：任务冻结与 veRL 接入准备](04-vanilla-grpo-v1-preparation-2026-07-21.md)。

## 证据与提交

| 内容 | 位置 |
|---|---|
| 运行时与版本手册 | `docs/grpo-runtime-setup.md` |
| Vanilla GRPO 配置 | `configs/verl/vanilla_grpo.yaml` |
| 运行时 preflight | `scripts/check_grpo_runtime.py` |
| 项目 AgentLoop | `src/shopping_grpo/verl_adapter/agent_loop.py` |
| ShopSimulator tool adapter | `src/shopping_grpo/verl_adapter/tools.py` |
| 第一轮失败日志 | `/root/autodl-tmp/grpo-smoke-logs-20260723/` |
| `bdec522` smoke 日志 | `/root/autodl-tmp/grpo-smoke-logs-20260723-bdec522/` |
| 96GB old-log-prob OOM 日志 | `/root/autodl-tmp/grpo-smoke-logs-20260723-96gb-8b6610d/` |
| 96GB bypass 成功日志 | `/root/autodl-tmp/grpo-smoke-logs-20260723-96gb-bypass/` |

关键提交：

- `f240429`：准备任务 split 与初版 veRL adapter；
- `cfb29c4`：加入 Vanilla GRPO runtime；
- `d5bfa52`：迁移到官方 veRL 0.8 AgentLoop；
- `bdec522`：使用 `lora.merge=true` 修复 Qwen3.5 rollout 权重同步；
- `679e09c`：为 veRL worker 接入纯 PyTorch padding fallback；
- `a14538f`：复用 vLLM rollout log-prob，绕过 actor old-log-prob 重算。
