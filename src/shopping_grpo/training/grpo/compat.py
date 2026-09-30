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


# 组级指标暂存：优势计算（先执行）写入，metrics 计算（后执行）读取注入。
# 两个包装器同进程（trainer actor），按步顺序配对。
_GROUP_STASH: dict[str, float] = {}


def _apply_group_metrics_from_advantage(utils_module):
    """包装 v1.utils.compute_advantage_for_multi_trajectories：算完优势后
    暂存组级统计（其入参 data 带 uid + token_level_scores）。"""
    _orig_adv = utils_module.compute_advantage_for_multi_trajectories

    def _adv_with_group_stats(data, *args, **kwargs):
        out = _orig_adv(data, *args, **kwargs)
        try:
            uid = data.non_tensor_batch.get("uid")
            if uid is not None:
                scores = data.batch["token_level_scores"].sum(-1)
                _GROUP_STASH.clear()
                _GROUP_STASH.update(
                    compute_group_stats(
                        scores.tolist(),
                        [str(u) for u in (uid.tolist() if hasattr(uid, "tolist") else list(uid))],
                    )
                )
        except Exception as exc:  # 观测指标失败不阻断训练，但日志留痕
            import logging

            logging.getLogger(__name__).warning("group stats from advantage failed: %s", exc)
        return out

    utils_module.compute_advantage_for_multi_trajectories = _adv_with_group_stats


def _apply_group_metrics_injection(metric_utils_module):
    """包装 metric_utils.compute_data_metrics：把暂存的组级统计注入 step
    metrics（随之进入 console 与 SwanLab）。metrics_batch 由 TensorDict 重建
    不带 uid，组统计只能经暂存通道进来。上游补齐组级指标后删除本补丁。"""
    _orig_metrics = metric_utils_module.compute_data_metrics

    def _with_group_metrics(batch, *args, **kwargs):
        metrics = _orig_metrics(batch, *args, **kwargs)
        if _GROUP_STASH:
            metrics.update(_GROUP_STASH)
            _GROUP_STASH.clear()
        return metrics

    metric_utils_module.compute_data_metrics = _with_group_metrics


def _apply_save_and_stop(trainer_base_module, flag_path=None):
    """SAVE_AND_STOP 标志文件：任意时刻触发，在下一个安全点保存并退出。

    用户操作：touch $SHOPPING_GRPO_ROOT/outputs/SAVE_AND_STOP
    两个检查点：步入口（step）与优势计算后（rollout 完成、梯度更新前）。
    rollout 不改权重，所以 rollout 中途触发时保存的就是上一个完整步的状态，
    最多损失当前步已生成的轨迹。标志处理后立即删除，防下次启动误触发。
    退出用 sys.exit(0)：checkpoint 已同步落盘，Ray 侧的 actor 报错是预期噪音。
    """
    import sys

    flag = flag_path or os.path.join(
        os.environ.get("SHOPPING_GRPO_ROOT", "/data/jyh-yyh/shopping-grpo-longhorizon"),
        "outputs",
        "SAVE_AND_STOP",
    )
    trainer_cls = trainer_base_module.PPOTrainer

    def _save_and_exit(self):
        import logging

        os.remove(flag)
        logging.getLogger(__name__).info(
            "SAVE_AND_STOP detected: saving checkpoint then exiting"
        )
        self._save_checkpoint()
        sys.exit(0)

    _orig_step = trainer_cls.step

    def _step_with_stop_check(self, *args, **kwargs):
        if os.path.exists(flag):
            _save_and_exit(self)
        return _orig_step(self, *args, **kwargs)

    trainer_cls.step = _step_with_stop_check

    _orig_compute_advantage = trainer_cls._compute_advantage

    def _advantage_with_stop_check(self, batch, metrics):
        out = _orig_compute_advantage(self, batch, metrics)
        if os.path.exists(flag):
            _save_and_exit(self)
        return out

    trainer_cls._compute_advantage = _advantage_with_stop_check


def patch_module_after_import(module_name, apply_fn):
    """注册"导入后补丁"：目标模块被业务代码正常导入时才应用 apply_fn。

    2026-09-30 06:39 崩溃教训：worker_process_setup_hook 里直接 import
    verl.trainer.*（trainer_base / metric_utils / v1.utils 均实测会触发
    CUDA 初始化）改变了 worker 的初始化时序，ref model FSDP 建组时报
    NCCL "Duplicate GPU detected"。此函数在 hook 期零导入，只挂 meta_path
    监听；模块首次加载发生在 Ray 的 per-worker 环境就绪之后，时序安全。
    宿主机可单测（不依赖 veRL）。
    """
    import importlib.abc
    import importlib.util
    import sys

    if module_name in sys.modules:  # 已导入（防 hook 晚于业务导入）：直接补
        apply_fn(sys.modules[module_name])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != module_name:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if spec is None:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                apply_fn(module)

            spec.loader.exec_module = exec_module  # 实例属性遮蔽类方法
            return spec

    sys.meta_path.insert(0, _Finder())


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

    纪律：hook 期只允许导入 attention_utils 这组最小集（run1 v3 实测安全），
    其余补丁一律经 patch_module_after_import 延迟，见其 docstring。
    """
    install_torch_padding_fallback()
    # 组级 reward 指标（run2 起）
    patch_module_after_import(
        "verl.trainer.ppo.v1.utils", _apply_group_metrics_from_advantage
    )
    patch_module_after_import(
        "verl.trainer.ppo.metric_utils", _apply_group_metrics_injection
    )
    # 任意时刻停训保存（run2 起）：touch outputs/SAVE_AND_STOP 触发
    patch_module_after_import("verl.trainer.ppo.v1.trainer_base", _apply_save_and_stop)
