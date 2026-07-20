#!/usr/bin/env python3
"""用目标模型的 chat template 预检 SFT JSONL 是否能产生有效 assistant 标签。"""

import argparse

from shopping_grpo.sft_training import IGNORE_INDEX, load_supervised_examples


def parse_args():
    parser = argparse.ArgumentParser(description="预检 SFT JSONL 的 chat-template 与 loss mask")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--input", required=True, help="待训练的 JSONL")
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--show-example", action="store_true", help="打印第一条样本的标签片段")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请执行：uv pip install -r requirements-sft.txt") from exc

    # Qwen3.5 的 chat template 属于 processor；文本工具轨迹仍只会产生 token ids。
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    examples, stats = load_supervised_examples(
        args.input,
        tokenizer=tokenizer,
        chat_template=processor,
        max_length=args.max_length,
    )
    print(stats)
    if not examples:
        raise SystemExit("没有可训练样本；请检查模型 chat template、tools 和 --max-length")
    if args.show_example:
        labels = [token for token in examples[0]["labels"] if token != IGNORE_INDEX]
        print("first_task_id=", examples[0]["task_id"])
        print("assistant_label_preview=", tokenizer.decode(labels[:512]))


if __name__ == "__main__":
    main()
