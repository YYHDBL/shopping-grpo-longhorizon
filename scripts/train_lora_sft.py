#!/usr/bin/env python3
"""对验收后的 Shopping tool-calling 数据进行最小 LoRA SFT。"""

import argparse
import json
from functools import partial
from pathlib import Path

from shopping_grpo.sft_training import load_supervised_examples


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Transformers + PEFT 执行 Shopping LoRA SFT")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--train", type=Path, required=True, help="训练 SFT JSONL")
    parser.add_argument("--validation", type=Path, default=None, help="可选验证 SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="LoRA adapter 输出目录")
    parser.add_argument("--max-length", type=int, default=8192)
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
    return parser.parse_args()


def _training_dependencies():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请执行：uv pip install -r requirements-sft.txt") from exc
    return torch, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


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
    args = parse_args()
    if args.max_length < 1 or args.epochs <= 0:
        raise SystemExit("--max-length 与 --epochs 必须为正数")
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = _training_dependencies()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_examples, train_stats = load_supervised_examples(
        args.train, tokenizer=tokenizer, max_length=args.max_length
    )
    print("train_data=", train_stats)
    if not train_examples:
        raise SystemExit("训练集没有可用样本；先运行 inspect_sft_data.py 排查")
    validation_examples = []
    if args.validation:
        validation_examples, validation_stats = load_supervised_examples(
            args.validation, tokenizer=tokenizer, max_length=args.max_length
        )
        print("validation_data=", validation_stats)
        if not validation_examples:
            raise SystemExit("验证集没有可用样本；请调整划分或 --max-length")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
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
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_torch_dataset(train_examples, torch),
        eval_dataset=_torch_dataset(validation_examples, torch) if validation_examples else None,
        data_collator=partial(_collate, pad_token_id=tokenizer.pad_token_id, torch=torch),
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "train_summary.json").write_text(
        json.dumps(
            {
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "train_loss": result.training_loss,
                "metrics": result.metrics,
                "arguments": vars(args),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"LoRA adapter 已保存到 {args.output}")


if __name__ == "__main__":
    main()
