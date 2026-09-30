import os


def compute_group_stats(scores: list[float], uids: list[str]) -> dict[str, float]:
    """组级 reward 统计（纯函数，宿主机可单测）。

    uid 形如 "<task-uuid>_<sample_idx>_<attempt>"（实测 run2 smoke 落盘：
    同任务 8 条采样的中间段递增 0..7），去掉末两段即组 ID；单成员组无组内
    方差，跳过。返回空 dict 表示没有可统计的组。
    """
    groups: dict[str, list[float]] = {}
    for score, uid in zip(scores, uids):
        groups.setdefault(str(uid).rsplit("_", 2)[0], []).append(float(score))
    stds = []
    for members in groups.values():
        if len(members) > 1:
            m = sum(members) / len(members)
            var = sum((x - m) ** 2 for x in members) / len(members)
            stds.append(var**0.5)
    if not stds:
        return {}
    return {
        "group/reward_std_mean": sum(stds) / len(stds),
        "group/zero_variance_ratio": sum(1 for s in stds if s <= 1e-9) / len(stds),
    }


def _patch_group_reward_metrics():
    """组级 reward 指标进 step metrics（veRL 只记全批次聚合，无组内统计）。

    注入点（2026-09-30 smoke 排障结论）：metrics_batch 由 TensorDict 重建，
    non_tensor_batch 不带 uid，直接在 compute_data_metrics 里拿不到组 ID。
    改为两段式：包装 compute_advantage_for_multi_trajectories（其入参 data
    带 uid + scores，且先于 metrics 执行）暂存组统计；compute_data_metrics
    包装器读暂存注入。worker_process_setup_hook 早于 trainer_base 的
    from-import 执行，两处包装均在绑定前落地。上游补齐组级指标后删除。
    """
    from verl.trainer.ppo import metric_utils as _mu
    from verl.trainer.ppo.v1 import utils as _vu

    _stash: dict[str, float] = {}

    _orig_adv = _vu.compute_advantage_for_multi_trajectories

    def _adv_with_group_stats(data, *args, **kwargs):
        out = _orig_adv(data, *args, **kwargs)
        try:
            uid = data.non_tensor_batch.get("uid")
            if uid is not None:
                scores = data.batch["token_level_scores"].sum(-1)
                _stash.clear()
                _stash.update(
                    compute_group_stats(
                        scores.tolist(),
                        [str(u) for u in (uid.tolist() if hasattr(uid, "tolist") else list(uid))],
                    )
                )
        except Exception as exc:  # 观测指标失败不阻断训练，但日志留痕
            import logging

            logging.getLogger(__name__).warning("group stats from advantage failed: %s", exc)
        return out

    _vu.compute_advantage_for_multi_trajectories = _adv_with_group_stats

    _orig_metrics = _mu.compute_data_metrics

    def _with_group_metrics(batch, *args, **kwargs):
        metrics = _orig_metrics(batch, *args, **kwargs)
        if _stash:
            metrics.update(_stash)
            _stash.clear()
        return metrics

    _mu.compute_data_metrics = _with_group_metrics


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。"""
    # Ray 分配 GPU 前执行此 hook；设备可用性查询必须避免初始化 CUDA。
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions


def _patch_save_and_stop():
    """SAVE_AND_STOP 标志文件：任意时刻触发，在下一个安全点保存并退出。

    用户操作：touch $SHOPPING_GRPO_ROOT/outputs/SAVE_AND_STOP
    两个检查点：步入口（step）与优势计算后（rollout 完成、梯度更新前）。
    rollout 不改权重，所以 rollout 中途触发时保存的就是上一个完整步的状态，
    最多损失当前步已生成的轨迹。标志处理后立即删除，防下次启动误触发。
    退出用 sys.exit(0)：checkpoint 已同步落盘，Ray 侧的 actor 报错是预期噪音。
    """
    import sys

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    flag = os.path.join(
        os.environ.get("SHOPPING_GRPO_ROOT", "/data/jyh-yyh/shopping-grpo-longhorizon"),
        "outputs",
        "SAVE_AND_STOP",
    )

    def _save_and_exit(self):
        import logging

        os.remove(flag)
        logging.getLogger(__name__).info(
            "SAVE_AND_STOP detected: saving checkpoint then exiting"
        )
        self._save_checkpoint()
        sys.exit(0)

    _orig_step = PPOTrainer.step

    def _step_with_stop_check(self, *args, **kwargs):
        if os.path.exists(flag):
            _save_and_exit(self)
        return _orig_step(self, *args, **kwargs)

    PPOTrainer.step = _step_with_stop_check

    _orig_compute_advantage = PPOTrainer._compute_advantage

    def _advantage_with_stop_check(self, batch, metrics):
        out = _orig_compute_advantage(self, batch, metrics)
        if os.path.exists(flag):
            _save_and_exit(self)
        return out

    PPOTrainer._compute_advantage = _advantage_with_stop_check


def install_worker_hooks():
    """worker_process_setup_hook 入口（容器内执行，run2 起配置指向这里）。

    宿主机 .venv 是老版 veRL（无顶层 DataProto），组级指标补丁只能在
    容器的 0.9.1 里导入；与注意力补丁拆开，宿主机单测只测后者。
    """
    install_torch_padding_fallback()
    # 组级 reward 指标（run2 起）：早于 trainer_base 的 from-import 生效
    _patch_group_reward_metrics()
    # 任意时刻停训保存（run2 起）：touch outputs/SAVE_AND_STOP 触发
    _patch_save_and_stop()
