"""Optional per-rank device binding and update receipts; leave checkpointing to veRL."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def install():
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()
    if not os.environ.get("SHOPPING_UPDATE_AUDIT_DIR"):
        return
    import torch
    import torch.distributed as dist
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if getattr(FSDPEngine, "_broad_instrumented", False):
        return
    FSDPEngine._broad_instrumented = True
    root = Path(os.environ["SHOPPING_UPDATE_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    original_step = FSDPEngine.optimizer_step
    original_initialize = FSDPEngine.initialize

    def initialize(engine):
        # Ray workers share physical visibility; explicitly select this worker's
        # assigned local GPU in the thread performing FSDP initialization.
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)
        append(
            f"device_rank{dist.get_rank()}.jsonl",
            {
                "rank": dist.get_rank(),
                "local_rank": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "current_device": torch.cuda.current_device(),
            },
        )
        return original_initialize(engine)

    def fingerprint(engine):
        h = hashlib.sha256()
        count = 0
        for name, p in engine.module.named_parameters():
            if not p.requires_grad:
                continue
            t = p.detach()
            if hasattr(t, "to_local"):
                t = t.to_local()
            t = t.cpu().contiguous()
            h.update(name.encode())
            h.update(t.view(torch.uint8).numpy().tobytes())
            count += t.numel()
        return {"sha256": h.hexdigest(), "local_trainable_elements": count}

    def append(name, record):
        with (root / name).open("a") as f:
            f.write(json.dumps(record) + "\n")

    def optimizer_step(engine):
        if shutil.disk_usage(root).free < 3 * 1024**3:
            raise RuntimeError("Less than 3 GiB free; stop without deleting data")
        before = fingerprint(engine)
        norm = original_step(engine)
        after = fingerprint(engine)
        engine._broad_steps = getattr(engine, "_broad_steps", 0) + 1
        rank = dist.get_rank()
        append(
            f"optimizer_rank{rank}.jsonl",
            {
                "rank": rank,
                "world_size": dist.get_world_size(),
                "optimizer_calls": engine._broad_steps,
                "time": time.time(),
                "grad_norm": float(norm),
                "before": before,
                "after": after,
                "parameters_changed": before["sha256"] != after["sha256"],
            },
        )
        if not torch.isfinite(torch.tensor(float(norm))):
            raise RuntimeError("Nonfinite gradient")
        return norm

    FSDPEngine.optimizer_step = optimizer_step
    FSDPEngine.initialize = initialize
