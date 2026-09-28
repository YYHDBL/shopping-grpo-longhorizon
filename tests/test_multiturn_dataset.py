# ShoppingMultiTurnSFTDataset CPU 测试：真实 tokenizer + 小型 parquet
# 覆盖：veRL custom_cls 加载、map 归一化、无空参数、finish reason enum、
# loss mask 覆盖 assistant 段、thinking 禁用边界、超长 error。
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_sft_parquet import arrow_schema, slim_messages, _fill  # noqa: E402
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402

MODEL = os.environ.get(
    "SHOP_MODEL", str(ROOT / "models/Qwen3.5-9B"))

SAMPLE = {
    "record_id": "100000000001",
    "teacher": "glm",
    "difficulty": "easy",
    "persona_condition": "no_profile",
    "tokens": 10,
    "messages": [
        {"role": "system", "train": 0, "content": "SYS"},
        {"role": "user", "train": 0, "content": "买一个测试商品"},
        {"role": "assistant", "train": 1, "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "search_products",
                "arguments": '{"query": "测试商品"}'}}]},
        {"role": "tool", "train": 0, "tool_call_id": "c1",
         "content": "搜索结果页"},
        {"role": "assistant", "train": 1, "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {
                "name": "finish_without_purchase",
                "arguments": '{"reason": "no_suitable_product"}'}}]},
    ],
}


def write_parquet(path, samples):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    tools_json = json.dumps(SHOP_TOOL_SCHEMAS, ensure_ascii=False)
    for s in samples:
        rows.append({
            "record_id": s["record_id"],
            "teacher": s.get("teacher"),
            "difficulty": s["difficulty"],
            "persona_condition": s["persona_condition"],
            "tokens": s.get("tokens", 10),
            "messages": slim_messages(s),
            "tools": tools_json,
            "enable_thinking": False,
        })
    schema = arrow_schema()
    types = {f.name: f.type for f in schema}
    table = pa.Table.from_pandas(
        pd.DataFrame([{n: _fill(r.get(n), types[n]) for n in types}
                      for r in rows]),
        schema=schema, preserve_index=False)
    pq.write_table(table, path)


@unittest.skipUnless(Path(MODEL).exists(), "Qwen3.5-9B 模型目录不存在")
class ShoppingMultiTurnDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        import tempfile
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        cls.tmp = tempfile.TemporaryDirectory()
        cls.parquet = Path(cls.tmp.name) / "mini.parquet"
        write_parquet(cls.parquet, [SAMPLE])
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, trust_remote_code=True)
        cls.config = OmegaConf.create({
            "max_length": 4096, "truncation": "error",
            "messages_key": "messages", "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
        })

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _dataset(self):
        from shopping_grpo.training.sft.multiturn_dataset import (
            ShoppingMultiTurnSFTDataset,
        )
        return ShoppingMultiTurnSFTDataset(
            parquet_files=[str(self.parquet)],
            tokenizer=self.tokenizer, config=self.config)

    def test_custom_cls_loads_via_verl(self):
        from verl.utils.import_utils import load_extern_object
        cls = load_extern_object(
            "pkg://shopping_grpo.training.sft.multiturn_dataset",
            "ShoppingMultiTurnSFTDataset")
        self.assertEqual(cls.__name__, "ShoppingMultiTurnSFTDataset")

    def test_tools_schema_roundtrip_keeps_finish_enum(self):
        ds = self._dataset()
        _, tools, _ = ds._load_row(0)
        self.assertEqual(tools, SHOP_TOOL_SCHEMAS)
        finish = next(t for t in tools
                      if t["function"]["name"] == "finish_without_purchase")
        self.assertEqual(
            finish["function"]["parameters"]["properties"]["reason"]["enum"],
            ["no_suitable_product"])

    def test_arguments_normalized_and_no_none_params(self):
        ds = self._dataset()
        messages, _, _ = ds._load_row(0)
        search = messages[2]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(search, {"query": "测试商品"})
        item = ds[0]
        trained = self.tokenizer.decode(
            item["input_ids"][item["loss_mask"].bool()].tolist())
        self.assertNotIn("None", trained)
        self.assertIn("<parameter=query>", trained)

    def test_loss_mask_covers_only_assistant_segments(self):
        item = self._dataset()[0]
        self.assertEqual(int(item["loss_mask"].sum()) > 0, True)
        ids = item["input_ids"].tolist()
        mask = item["loss_mask"].bool().tolist()
        first_trained = self.tokenizer.decode(
            [t for t, m in zip(ids, mask) if m][:6])
        self.assertIn("tool_call", first_trained)

    def test_thinking_disabled_no_think_tags(self):
        item = self._dataset()[0]
        trained = self.tokenizer.decode(
            item["input_ids"][item["loss_mask"].bool()].tolist())
        self.assertNotIn("<think>", trained)
        self.assertNotIn("</think>", trained)

    def test_overlong_raises_with_truncation_error(self):
        from omegaconf import OmegaConf
        from shopping_grpo.training.sft.multiturn_dataset import (
            ShoppingMultiTurnSFTDataset,
        )
        config = OmegaConf.create({**self.config, "max_length": 16})
        ds = ShoppingMultiTurnSFTDataset(
            parquet_files=[str(self.parquet)],
            tokenizer=self.tokenizer, config=config)
        with self.assertRaises(ValueError):
            ds[0]


if __name__ == "__main__":
    unittest.main()
