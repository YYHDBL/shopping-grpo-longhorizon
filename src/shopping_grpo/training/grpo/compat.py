"""veRL 0.8 的窄范围运行时兼容。"""


def _install_trace_actor_update():
    """Score private gold targets before the existing veRL actor update."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._update_actor, "_shopping_trace", False):
        return

    from shopping_grpo.training.grpo.trace import (
        apply_trace_advantages,
        build_trace_score_batch,
    )

    original_update_actor = RayPPOTrainer._update_actor

    def update_actor_with_trace(self, batch):
        config = self.config.get("shopping_trace", {})
        enabled = bool(config.get("enable", False))
        if not enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            return original_update_actor(self, batch)
        if not self.ref_in_actor:
            raise RuntimeError("shopping TRACE requires LoRA so the frozen base model is available")

        score_batch, state_counts = build_trace_score_batch(
            batch,
            self.tokenizer,
            max_sequence_length=int(config["max_sequence_length"]),
        )
        reference = self._compute_ref_log_prob(score_batch)
        target_mask = score_batch.batch["response_mask"]
        mean_log_probs = (
            reference.batch["ref_log_prob"] * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1)
        batch.non_tensor_batch.pop("trace_target", None)
        trace_metrics = apply_trace_advantages(
            batch,
            mean_log_probs,
            state_counts,
            epsilon=float(config["epsilon"]),
            horizon=int(config["horizon"]),
            discount=float(config["discount"]),
            terminal_weight=float(config["terminal_weight"]),
            outcome_weight=float(config["outcome_weight"]),
            turn_weight=float(config["turn_weight"]),
        )
        trace_metrics["trace/target_log_prob_mean"] = float(mean_log_probs.mean())
        output = original_update_actor(self, batch)
        output.meta_info["metrics"].update(trace_metrics)
        return output

    update_actor_with_trace._shopping_trace = True
    RayPPOTrainer._update_actor = update_actor_with_trace


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。"""
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions
    _install_trace_actor_update()
