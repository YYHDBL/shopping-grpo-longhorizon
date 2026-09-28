# SFT 教材转 veRL 训练格式：samples.jsonl -> train.parquet
# 列约定见 multiturn_dataset.py 头注：
#   messages 用 arrow struct + map（无参数不补 None），逐消息保留 train 标记
#   （守卫拒绝的 assistant 段 train=0，数据集据此置 loss mask=0）；
#   tools 存完整 schema 的 JSON 字符串（保留 enum）；
#   enable_thinking 恒为 False（全链路关思考）；
#   准入审计列（accept_reason/reward_type/reward_valid/over/termination_reason/
#   jev_verdict/purchase）随行落盘，替代购买的准入依据可独立追溯。
# 一律 fail fast，禁止静默转换：非法 JSON 参数、null/空字符串参数值、
# token 统计缺失都立即报错退出（历史教训：把 None 静默转成空字符串产生了
# view_features({"__dummy__": ""}) 一类参数语义污染）。
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
        ("train", pa.int64()),
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
        ("accept_reason", pa.string()),
        ("reward_type", pa.string()),
        ("reward_valid", pa.bool_()),
        ("over", pa.bool_()),
        ("termination_reason", pa.string()),
        ("jev_verdict", pa.string()),
        ("purchase", pa.string()),
    ])


def parse_arguments(record_id, tool_call):
    """解析并校验一条工具调用的参数；非法立即报错，绝不静默修复。"""
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw = function.get("arguments")
    if not isinstance(raw, str):
        raise ValueError(
            f"{record_id}: {name} arguments must be a JSON string, "
            f"got {type(raw).__name__}"
        )
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{record_id}: {name} arguments is not valid JSON: {raw!r}"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(
            f"{record_id}: {name} arguments must decode to an object, "
            f"got {type(arguments).__name__}"
        )
    for key, value in arguments.items():
        if value is None:
            raise ValueError(f"{record_id}: {name} parameter {key!r} is null")
        if isinstance(value, str) and not value:
            raise ValueError(f"{record_id}: {name} parameter {key!r} is empty")
    return name, arguments


def slim_messages(sample):
    record_id = sample["record_id"]
    messages = []
    for m in sample["messages"]:
        role = m["role"]
        if role == "assistant":
            tool_calls = None
            for call in m.get("tool_calls") or []:
                name, arguments = parse_arguments(record_id, call)
                call = dict(call)
                call["function"] = {
                    "name": name,
                    "arguments": arguments,
                }
                (tool_calls := tool_calls or []).append(call)
            messages.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "train": int(m.get("train", 1)),
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": m.get("content") or "",
                "train": int(m.get("train", 0)),
            })
        else:
            messages.append({
                "role": role,
                "content": m.get("content") or "",
                "train": int(m.get("train", 0)),
            })
    return messages


def _fill(value, arrow_type):
    if isinstance(arrow_type, pa.MapType):
        value_type = arrow_type.item_type
        result = {}
        for k, v in (value or {}).items():
            if value_type == pa.string():
                text = "" if v is None else str(v)
                if not text:
                    raise ValueError(f"empty map value for key {k!r}")
                result[str(k)] = text
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
    missing = [s["record_id"] for s in samples if s["record_id"] not in tokens]
    if missing:
        raise SystemExit(
            f"token 统计缺失 {len(missing)} 条（先运行 scripts/token_stats.py）："
            + ", ".join(missing[:10])
        )
    before = len(samples)
    samples = [s for s in samples if tokens[s["record_id"]] <= args.max_tokens]
    dropped = before - len(samples)

    tools_json = json.dumps(SHOP_TOOL_SCHEMAS, ensure_ascii=False)
    rows = []
    for s in samples:
        rows.append({
            "record_id": s["record_id"],
            "teacher": s.get("teacher"),
            "difficulty": s["difficulty"],
            "persona_condition": s["persona_condition"],
            "tokens": tokens[s["record_id"]],
            "messages": slim_messages(s),
            "tools": tools_json,
            "enable_thinking": False,
            "accept_reason": s.get("accept_reason"),
            "reward_type": s.get("reward_type"),
            "reward_valid": s.get("reward_valid"),
            "over": s.get("over"),
            "termination_reason": s.get("termination_reason"),
            "jev_verdict": (
                json.dumps(s["jev_verdict"], ensure_ascii=False)
                if s.get("jev_verdict") else None
            ),
            "purchase": (
                json.dumps(s["purchase"], ensure_ascii=False)
                if s.get("purchase") else None
            ),
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
