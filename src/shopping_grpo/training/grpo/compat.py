import os


def _patch_group_reward_metrics():
    """组级 reward 指标进 step metrics（veRL 只记全批次聚合，无组内统计）。

    worker_process_setup_hook 早于 trainer_base 的 from-import 执行，
    包装 compute_data_metrics 后，group/reward_std_mean 与
    group/zero_variance_ratio 随原 dict 进入 console + SwanLab。
    上游补齐组级指标后删除此补丁。
    """
    from verl.trainer.ppo import metric_utils as _mu

    _orig = _mu.compute_data_metrics

    def _with_group_metrics(batch, *args, **kwargs):
        metrics = _orig(batch, *args, **kwargs)
        try:
            scores = batch.batch["token_level_scores"].sum(-1)
            uid = batch.non_tensor_batch.get("uid")
            if uid is None:
                return metrics
            groups: dict[str, list[float]] = {}
            for i, g in enumerate(uid.tolist() if hasattr(uid, "tolist") else list(uid)):
                groups.setdefault(str(g), []).append(float(scores[i]))
            stds = []
            for members in groups.values():
                if len(members) > 1:
                    m = sum(members) / len(members)
                    var = sum((x - m) ** 2 for x in members) / len(members)
                    stds.append(var**0.5)
            if stds:
                metrics["group/reward_std_mean"] = sum(stds) / len(stds)
                metrics["group/zero_variance_ratio"] = (
                    sum(1 for s in stds if s <= 1e-9) / len(stds)
                )
        except Exception as exc:  # 观测指标失败不阻断训练，但日志留痕
            import logging

            logging.getLogger(__name__).warning("group metrics failed: %s", exc)
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


def install_worker_hooks():
    """worker_process_setup_hook 入口（容器内执行，run2 起配置指向这里）。

    宿主机 .venv 是老版 veRL（无顶层 DataProto），组级指标补丁只能在
    容器的 0.9.1 里导入；与注意力补丁拆开，宿主机单测只测后者。
    """
    install_torch_padding_fallback()
    # 组级 reward 指标（run2 起）：早于 trainer_base 的 from-import 生效
    _patch_group_reward_metrics()
