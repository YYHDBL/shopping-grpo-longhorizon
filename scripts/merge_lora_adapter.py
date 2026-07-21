#!/usr/bin/env python3
"""把完成 SFT 的 LoRA adapter 合并为 GRPO 的独立起点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose_model_class(config, causal_model_class, multimodal_model_class):
    """Qwen3.5 走官方多模态类；其他 CausalLM 保持普通路径。"""
    return multimodal_model_class if str(getattr(config, "model_type", "")).startswith("qwen3_5") else causal_model_class


def build_merge_manifest(base_model, adapter_path, output_path, model_type):
    """输出可审计清单；GRPO 必须新挂 adapter，不能覆盖这份 checkpoint。"""
    return {
        "operation": "peft_merge_and_unload",
        "source": {"base_model": str(base_model), "adapter": str(adapter_path), "model_type": str(model_type)},
        "output": str(output_path),
        "next_step": "load this standalone checkpoint as GRPO base and attach a new LoRA adapter",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="合并 LoRA SFT adapter，为 GRPO 创建独立 BF16 起点")
    parser.add_argument("--base-model", required=True, help="与 SFT 完全一致的原始模型路径或 Hugging Face 名称")
    parser.add_argument("--adapter", type=Path, required=True, help="SFT LoRA adapter 目录")
    parser.add_argument("--output", type=Path, required=True, help="新的 merged checkpoint 目录，必须为空")
    parser.add_argument("--bf16", action="store_true", help="以 bf16 合并；4090/RTX PRO 6000 建议开启")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{args.output}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit("缺少合并依赖；请在 SFT/GRPO 虚拟环境执行：uv pip install -r requirements-sft.txt") from exc

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_class = choose_model_class(config, AutoModelForCausalLM, AutoModelForMultimodalLM)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"加载 base={args.base_model} model_type={config.model_type} dtype={dtype}")
    base = model_class.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True, max_shard_size=args.max_shard_size)
    AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(str(args.output))
    manifest = build_merge_manifest(args.base_model, args.adapter, args.output, config.model_type)
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
