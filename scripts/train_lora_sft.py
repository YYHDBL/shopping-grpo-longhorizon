#!/usr/bin/env python3
"""对验收后的 Shopping tool-calling 数据进行最小 LoRA SFT。"""

import argparse
import json
import time as _time
from functools import partial
from pathlib import Path

from transformers import TrainerCallback

from shopping_grpo.sft_training import load_supervised_examples


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen3.5 的大多数文本层是 Gated DeltaNet，不能遗漏其线性注意力投影。
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Transformers + PEFT 执行 Shopping LoRA SFT")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--train", type=Path, required=True, help="训练 SFT JSONL")
    parser.add_argument("--validation", type=Path, default=None, help="可选验证 SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="LoRA adapter 输出目录")
    # 24k 可保留当前真实轨迹的约 93%，48G 显存配合 batch=1 与梯度检查点可稳定训练。
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--bf16", action="store_true", help="使用 bf16；支持的 CUDA GPU 建议开启")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=-1, help="最大训练步数（-1=完整 epoch）；用于冒烟测试")
    parser.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 记录")
    parser.add_argument("--wandb-project", default="shopping-grpo", help="W&B project 名")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run 名；默认自动生成")
    return parser.parse_args()


def _training_dependencies():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请执行：uv pip install -r requirements-sft.txt") from exc
    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )


def _load_preprocessing_components(model_name, auto_config, auto_tokenizer, auto_processor):
    """按模型配置选择 chat template 的持有者。

    Qwen3.5 是带视觉编码器的条件生成模型，官方模板由 processor 提供；本项目
    当前数据仅含文本和工具调用，因此 labels 仍用 processor.tokenizer 的 token id。
    其他纯文本因果模型保持原来的 tokenizer 路径。
    """
    config = auto_config.from_pretrained(model_name, trust_remote_code=True)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = auto_processor.from_pretrained(model_name, trust_remote_code=True)
        return processor.tokenizer, processor, True
    tokenizer = auto_tokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, tokenizer, False


def _torch_dataset(examples, torch):
    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            return {
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(example["labels"], dtype=torch.long),
            }

    return TokenizedDataset()


def _collate(batch, pad_token_id, torch):
    """右侧 padding，labels 的 padding 永远不参与 loss。"""
    max_length = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    _start_time = _time.time()
    args = parse_args()
    if args.max_length < 1 or args.epochs <= 0:
        raise SystemExit("--max-length 与 --epochs 必须为正数")
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = _training_dependencies()

    # --- Progress callback: 按步输出 loss / 显存 / 耗时 ---
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.step_start = None
            self.epoch_start = None
            self.step_count = 0
        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = _time.time()
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % args.logging_steps != 0 and not state.is_world_process_zero:
                return
            elapsed = _time.time() - self.step_start if self.step_start else 0
            loss = float("nan")
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry and entry["loss"] is not None:
                        loss = float(entry["loss"])
                        break
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            eta_seconds = (state.max_steps - state.global_step) * elapsed if state.global_step > 0 else 0
            eta_str = f"{eta_seconds/60:.0f}min" if eta_seconds > 0 else "?"
            print(f"[step {state.global_step}/{state.max_steps}] loss={loss:.4f}  step_t={elapsed:.1f}s  GPU={gpu_mem:.1f}GiB  ETA={eta_str}")
            self.step_count = state.global_step
            if _wandb_enabled:
                try:
                    _wandb.log({"loss": loss, "gpu_memory_gib": gpu_mem, "step_time_s": elapsed}, step=state.global_step)
                except Exception:
                    pass
        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start = _time.time()
            print(f"\n{'='*60}\n  EPOCH {int(state.epoch)} 开始  steps={state.max_steps}\n{'='*60}")
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_time = _time.time() - self.epoch_start if self.epoch_start else 0
            print(f"  EPOCH {int(state.epoch)} 完成  耗时={epoch_time/60:.1f}min")

    tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
        args.model,
        auto_config=AutoConfig,
        auto_tokenizer=AutoTokenizer,
        auto_processor=AutoProcessor,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Phase 1: 加载训练数据 ----
    print(f"\n{'='*60}")
    print(f"  Phase 1/3: 加载 & Tokenize 训练数据 (max_length={args.max_length})")
    print(f"{'='*60}")
    train_examples, train_stats = load_supervised_examples(
        args.train,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=args.max_length,
    )
    print("train_data=", train_stats)
    if not train_examples:
        raise SystemExit("训练集没有可用样本；先运行 inspect_sft_data.py 排查")
    validation_examples = []
    if args.validation:
        validation_examples, validation_stats = load_supervised_examples(
            args.validation,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=args.max_length,
        )
        print("validation_data=", validation_stats)
        if not validation_examples:
            raise SystemExit("验证集没有可用样本；请调整划分或 --max-length")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM

    # ---- Phase 2: 加载模型 + LoRA ----
    print(f"\n{'='*60}")
    print(f"  Phase 2/3: 加载模型 Qwen/Qwen3.5-2B + LoRA (r={args.lora_r})")
    print(f"{'='*60}")
    model = model_class.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(args.target_modules),
        ),
    )
    model.print_trainable_parameters()

    # --- wandb 初始化 ---
    if args.wandb:
        run_name = args.wandb_run_name or f"lora-r{args.lora_r}-bs{args.per_device_train_batch_size}x{args.gradient_accumulation_steps}-lr{args.learning_rate}"
        try:
            import wandb as _wandb
            # 先尝试在线模式；没登录或网络不通则自动 fallback 到 offline
            _wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
                        settings=_wandb.Settings(init_timeout=10))
            if _wandb.run.mode == "offline":
                print(f"[W&B] 离线模式 — 数据保存在本地。训练完成后运行:  wandb sync {_wandb.run.dir}")
            else:
                print(f"[W&B] 已连接 → {_wandb.run.get_url()}")
            _wandb_enabled = True
        except Exception as exc:
            print(f"[W&B] 初始化失败: {exc}")
            print(f"[W&B] 尝试离线模式...")
            try:
                _wandb.init(project=args.wandb_project, name=run_name, config=vars(args), mode="offline")
                _wandb_enabled = True
                print(f"[W&B] 离线模式已启用，数据保存在本地")
            except Exception:
                _wandb_enabled = False
                print(f"[W&B] 离线模式也失败，跳过 W&B 记录")
    else:
        _wandb_enabled = False

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if validation_examples else "no",
        report_to="wandb" if _wandb_enabled else "none",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_torch_dataset(train_examples, torch),
        eval_dataset=_torch_dataset(validation_examples, torch) if validation_examples else None,
        data_collator=partial(_collate, pad_token_id=tokenizer.pad_token_id, torch=torch),
        callbacks=[ProgressCallback()],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    chat_template.save_pretrained(str(args.output))

    # --- 训练完成摘要 ---
    total_time = _time.time() - _start_time
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    train_summary = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_loss": result.training_loss,
        "metrics": result.metrics,
        "peak_gpu_memory_gib": round(gpu_peak, 2),
        "total_time_minutes": round(total_time / 60, 1) if total_time else None,
        "arguments": vars(args),
    }

    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"  train_loss={result.training_loss:.4f}")
    print(f"  eval_loss={result.metrics.get('eval_loss', 'N/A')}")
    print(f"  peak_gpu={gpu_peak:.1f} GiB")
    print(f"  adapter → {args.output}")
    print(f"{'='*60}\n")

    (args.output / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if _wandb_enabled:
        try:
            _wandb.log({"final_train_loss": result.training_loss, "peak_gpu_gib": gpu_peak})
            _wandb.finish()
        except Exception:
            pass
    print(f"LoRA adapter 已保存到 {args.output}")


if __name__ == "__main__":
    main()
