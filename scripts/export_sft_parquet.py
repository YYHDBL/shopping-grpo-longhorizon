# SFT 教材转 veRL 训练格式：samples.jsonl -> train.parquet
# veRL MultiTurnSFTDataset 约定：messages 列（OpenAI 风格，逐条渲染）+ tools 列；
# loss mask 由 verl 计算（assistant 训练、其余遮蔽），与教材加工规则一致。
# 附带 CPU 抽验：用模型 tokenizer 渲染一条样本，验证 loss mask 只落在 assistant token。
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


def arrow_schema():
    # arguments 与 properties 用 map 类型：不同工具键不同、空 map 可写、
    # 不需要补 None（补 None 会把 query=x 渲染成 asin=null 的污染参数）。
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
    properties = pa.map_(pa.string(), pa.struct([("type", pa.string())]))
    parameters = pa.struct([
        ("type", pa.string()),
        ("properties", properties),
        ("required", pa.list_(pa.string())),
        ("additionalProperties", pa.bool_()),
    ])
    tool = pa.struct([
        ("type", pa.string()),
        ("function", pa.struct([
            ("name", pa.string()), ("description", pa.string()),
            ("parameters", parameters),
        ])),
    ])
    return pa.schema([
        ("record_id", pa.string()),
        ("teacher", pa.string()),
        ("difficulty", pa.string()),
        ("persona_condition", pa.string()),
        ("chars", pa.int64()),
        ("messages", pa.list_(message)),
        ("tools", pa.list_(tool)),
        ("enable_thinking", pa.bool_()),
    ])


def _fill(value, arrow_type):
    # 递归把 python dict 补齐 arrow struct 声明的全部字段（缺失即 None）；
    # map 类型直接透传（不补键，避免污染参数）。
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
        return [_fill(item, arrow_type.value_field.type) for item in (value or [])]
    return value


def coerce_rows(rows):
    schema = arrow_schema()
    types = {f.name: f.type for f in schema}
    return [
        {name: _fill(row.get(name), types[name]) for name in types}
        for row in rows
    ]


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
                # 空 dict 保留（map 类型可写空值；None 会让模板渲染出错）
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
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--verify", type=int, default=3,
                        help="实例化 veRL 数据集并逐条验证前 N 条（CPU）")
    parser.add_argument("--verify-max-length", type=int, default=32768,
                        help="验证时数据集的 max_length")
    args = parser.parse_args()

    samples = [json.loads(l) for l in (OUT / "samples.jsonl").open(encoding="utf-8")]
    if args.max_chars:
        before = len(samples)
        samples = [s for s in samples if s["chars"] <= args.max_chars]
        print(f"超长丢弃：{before - len(samples)} 条，保留 {len(samples)}")

    rows = []
    for s in samples:
        rows.append({
            "record_id": s["record_id"],
            "teacher": s.get("teacher"),
            "difficulty": s["difficulty"],
            "persona_condition": s["persona_condition"],
            "chars": s["chars"],
            "messages": slim_messages(s),
            "tools": SHOP_TOOL_SCHEMAS,
            "enable_thinking": False,
        })
    schema = arrow_schema()
    table = pa.Table.from_pandas(
        pd.DataFrame(coerce_rows(rows)), schema=schema, preserve_index=False)
    out_path = OUT / "train.parquet"
    pq.write_table(table, out_path)
    print(f"写出 {table.num_rows} 条 -> {out_path}")
    print("列:", table.schema.names)

    if args.verify > 0:
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
        tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / "models/Qwen3.5-9B"), trust_remote_code=True)
        config = OmegaConf.create({
            "max_length": args.verify_max_length,
            "truncation": "error",
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
        })
        dataset = MultiTurnSFTDataset(
            parquet_files=[str(out_path)], tokenizer=tokenizer, config=config)
        print(f"\n数据集实例化成功，共 {len(dataset)} 条；逐条验证前 {args.verify} 条：")
        for i in range(min(args.verify, len(dataset))):
            item = dataset[i]
            input_ids = item["input_ids"]
            loss_mask = item["loss_mask"]
            n_train = int((loss_mask > 0).sum())
            print(f"  #{i}：{len(input_ids)} token，参与训练 token {n_train}"
                  f"（占比 {n_train / len(input_ids):.1%}），"
                  f"mask 与序列等长：{len(loss_mask) == len(input_ids)}")
            decoded = tokenizer.decode(input_ids[loss_mask.bool()][:40].tolist())
            print(f"     首个训练片段：{decoded[:80]!r}")


if __name__ == "__main__":
    main()
