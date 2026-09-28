# SFT 教材转 veRL 训练格式：samples.jsonl -> train.parquet
# 列约定见 multiturn_dataset.py 头注：
#   messages 用 arrow struct + map（无参数不补 None）；
#   tools 存完整 schema 的 JSON 字符串（保留 enum）；
#   enable_thinking 恒为 False（全链路关思考）。
# 超长按 token_stats.jsonl 的真实 token 数过滤（默认 16384，禁止截断）。
# 验证与训练走同一条代码路径：ShoppingMultiTurnSFTDataset + veRL load_extern_object。
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402

OUT = ROOT / "outputs/sft_dataset"
DEFAULT_MAX_TOKENS = 16384


def arrow_schema():
    arguments = pa.map_(pa.string(), pa.string())
    function = pa.struct([
        ("name", pa.string()), ("arguments", arguments),
    ])
    tool_call = pa.struct([
        ("id", pa.string()), ("type", pa.string()), ("function", function),
    ])
    message = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
        ("tool_calls", pa.list_(tool_call)),
        ("tool_call_id", pa.string()),
    ])
    return pa.schema([
        ("record_id", pa.string()),
        ("teacher", pa.string()),
        ("difficulty", pa.string()),
        ("persona_condition", pa.string()),
        ("tokens", pa.int64()),
        ("messages", pa.list_(message)),
        ("tools", pa.string()),
        ("enable_thinking", pa.bool_()),
    ])


def _fill(value, arrow_type):
    if isinstance(arrow_type, pa.MapType):
        value_type = arrow_type.item_type
        result = {}
        for k, v in (value or {}).items():
            if value_type == pa.string():
                result[str(k)] = "" if v is None else str(v)
            else:
                result[str(k)] = _fill(v, value_type)
        return result
    if isinstance(arrow_type, pa.StructType):
        result = {}
        value = value or {}
        for field in arrow_type:
            result[field.name] = _fill(value.get(field.name), field.type)
        return result
    if isinstance(arrow_type, pa.ListType):
        return [_fill(item, arrow_type.value_field.type)
                for item in (value or [])]
    return value


def slim_messages(sample):
    messages = []
    for m in sample["messages"]:
        role = m["role"]
        if role == "assistant":
            tool_calls = None
            for call in m.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    arguments = {}
                call = dict(call)
                call["function"] = {
                    "name": function.get("name"),
                    "arguments": arguments,
                }
                (tool_calls := tool_calls or []).append(call)
            messages.append({
                "role": "assistant",
                "content": m.get("content") or "",
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": m.get("content") or "",
            })
        else:
            messages.append({"role": role, "content": m.get("content") or ""})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="超长丢弃阈值（真实 token 数），禁止截断")
    parser.add_argument("--verify", type=int, default=3,
                        help="经训练同一条数据集路径验证前 N 条（CPU）")
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3.5-9B"),
                        help="tokenizer/模型目录（可用 SHOP_MODEL 环境变量覆盖）")
    args = parser.parse_args()
    import os
    args.model = os.environ.get("SHOP_MODEL", args.model)

    samples = [json.loads(l) for l in (OUT / "samples.jsonl").open(encoding="utf-8")]
    tokens = {json.loads(l)["record_id"]: json.loads(l)["tokens"]
              for l in (OUT / "token_stats.jsonl").open(encoding="utf-8")}
    before = len(samples)
    samples = [s for s in samples
               if tokens.get(s["record_id"], 0) <= args.max_tokens]
    dropped = before - len(samples)

    tools_json = json.dumps(SHOP_TOOL_SCHEMAS, ensure_ascii=False)
    rows = []
    for s in samples:
        rows.append({
            "record_id": s["record_id"],
            "teacher": s.get("teacher"),
            "difficulty": s["difficulty"],
            "persona_condition": s["persona_condition"],
            "tokens": tokens.get(s["record_id"]),
            "messages": slim_messages(s),
            "tools": tools_json,
            "enable_thinking": False,
        })
    schema = arrow_schema()
    types = {f.name: f.type for f in schema}
    table = pa.Table.from_pandas(
        pd.DataFrame([{name: _fill(row.get(name), types[name])
                       for name in types} for row in rows]),
        schema=schema, preserve_index=False)
    out_path = OUT / "train.parquet"
    pq.write_table(table, out_path)
    print(f"超长丢弃 {dropped} 条（>{args.max_tokens} token），"
          f"保留 {table.num_rows} 条 -> {out_path}")

    if args.verify > 0:
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from verl.utils.import_utils import load_extern_object
        dataset_cls = load_extern_object(
            "pkg://shopping_grpo.training.sft.multiturn_dataset",
            "ShoppingMultiTurnSFTDataset")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True)
        config = OmegaConf.create({
            "max_length": args.max_tokens,
            "truncation": "error",
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
        })
        dataset = dataset_cls(
            parquet_files=[str(out_path)], tokenizer=tokenizer, config=config)
        print(f"训练路径数据集实例化成功，共 {len(dataset)} 条；"
              f"验证前 {args.verify} 条：")
        for i in range(min(args.verify, len(dataset))):
            item = dataset[i]
            n_train = int(item["loss_mask"].sum())
            first = tokenizer.decode(
                item["input_ids"][item["loss_mask"].bool()][:20].tolist())
            print(f"  #{i}：{len(item['input_ids'])} token，"
                  f"训练 token {n_train}（{n_train/len(item['input_ids']):.1%}），"
                  f"首个训练片段 {first[:50]!r}")


if __name__ == "__main__":
    main()
