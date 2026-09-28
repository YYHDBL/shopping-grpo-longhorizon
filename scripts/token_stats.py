# 全量真实 token 统计（整序列渲染一次，用于确定 max_length 与超长过滤）
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_sft_parquet import slim_messages  # noqa: E402
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/Qwen3.5-9B"), trust_remote_code=True)
    samples = [json.loads(l) for l in
               (ROOT / "outputs/sft_dataset/samples.jsonl").open(encoding="utf-8")]
    out = ROOT / "outputs/sft_dataset/token_stats.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            messages = slim_messages(s)
            # 归一化：与训练数据集读回后的形态一致
            for m in messages:
                for call in m.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    if isinstance(fn.get("arguments"), list):
                        fn["arguments"] = dict(fn["arguments"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, tools=SHOP_TOOL_SCHEMAS,
                add_generation_prompt=False)
            n = len(tokenizer(text, add_special_tokens=False).input_ids)
            f.write(json.dumps({"record_id": s["record_id"], "tokens": n}) + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(samples)}", flush=True)
    print("done ->", out)

if __name__ == "__main__":
    main()
