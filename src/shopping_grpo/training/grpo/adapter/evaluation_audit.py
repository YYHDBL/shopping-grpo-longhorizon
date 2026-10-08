"""Optional evaluation-only device/loading audit. Configure EVAL_* paths explicitly."""

import hashlib
import json
import os
import time
from pathlib import Path


def install():
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

    install_torch_padding_fallback()
    import torch
    import torch.distributed as dist
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    if getattr(FSDPEngine, "_final200_eval_audit", False):
        return
    FSDPEngine._final200_eval_audit = True
    root = Path(os.environ["EVAL_CONDITION_DIR"])
    step = int(os.environ["EVAL_CHECKPOINT_STEP"])
    original_init = FSDPEngine.initialize
    original_load = FSDPEngine.load_checkpoint

    def fingerprint(engine):
        h = hashlib.sha256()
        count = 0
        bnz = 0
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
            if "lora_B" in name:
                bnz += int(torch.count_nonzero(t))
        return {"sha256": h.hexdigest(), "local_trainable_elements": count, "lora_b_nonzero": bnz}

    def record(kind, data):
        (root / (kind + "_rank" + str(dist.get_rank()) + ".json")).write_text(
            json.dumps(dict(data, rank=dist.get_rank(), time=time.time()), indent=2)
        )

    def initialize(engine):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        result = original_init(engine)
        fp = fingerprint(engine)
        record("initialized", dict(fp, device=torch.cuda.current_device(), step=step))
        if step == 0:
            assert fp["lora_b_nonzero"] == 0, "Baseline LoRA must be a zero delta"
        return result

    def load(engine, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        expected = str(
            Path(os.environ["EVAL_TRAINING_DIR"])
            / "checkpoints"
            / ("global_step_" + str(step))
            / "actor"
        )
        assert str(local_path) == expected and not del_local_after_load, (
            local_path,
            expected,
            del_local_after_load,
        )
        result = original_load(engine, local_path, hdfs_path, False, **kwargs)
        fp = fingerprint(engine)
        history = Path(os.environ["EVAL_TRAINING_DIR"]) / (
            "optimizer_rank" + str(dist.get_rank()) + ".jsonl"
        )
        matches = [
            json.loads(line)
            for line in history.read_text().splitlines()
            if json.loads(line)["optimizer_calls"] == step
        ]
        assert len(matches) == 1
        exp = matches[0]["after"]
        ok = all(fp[k] == exp[k] for k in ["sha256", "local_trainable_elements"])
        record("loaded", dict(fp, expected=exp, match=ok, checkpoint=str(local_path), step=step))
        assert ok, "Loaded adapter fingerprint differs from original training receipt"
        return result

    def reject_update(engine, *args, **kwargs):
        record("forbidden_optimizer_call", {"step": step})
        raise RuntimeError("Evaluation-only: optimizer step forbidden")

    def reject_save(engine, *args, **kwargs):
        raise RuntimeError("Evaluation-only: checkpoint save forbidden")

    FSDPEngine.initialize = initialize
    FSDPEngine.load_checkpoint = load
    FSDPEngine.optimizer_step = reject_update
    FSDPEngine.save_checkpoint = reject_save
