"""Qwen3.5 使用 qwen3_coder XML 工具协议，不能按 Hermes JSON 解析。"""

import asyncio
import json
import unittest

from shopping_grpo.verl_adapter.qwen3_coder_parser import Qwen3CoderToolParser


class FakeTokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, token_ids, **kwargs):
        del token_ids, kwargs
        return self.text


class Qwen3CoderParserTest(unittest.TestCase):
    def test_extracts_xml_function_and_multiline_parameters(self):
        text = """I should search first.
<tool_call>
<function=search_products>
<parameter=query>
blue ceramic
mug
</parameter>
</function>
</tool_call>"""

        async def run():
            parser = Qwen3CoderToolParser(FakeTokenizer(text))
            content, calls = await parser.extract_tool_calls([1, 2, 3])
            self.assertEqual(content, "I should search first.")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].name, "search_products")
            self.assertEqual(json.loads(calls[0].arguments), {"query": "blue ceramic\nmug"})

        asyncio.run(run())

    def test_extracts_empty_and_parallel_calls_without_eval(self):
        text = """<tool_call><function=view_features></function></tool_call>
<tool_call><function=back_to_search></function></tool_call>"""

        async def run():
            parser = Qwen3CoderToolParser(FakeTokenizer(text))
            content, calls = await parser.extract_tool_calls([1])
            self.assertEqual(content, "")
            self.assertEqual([call.name for call in calls], ["view_features", "back_to_search"])
            self.assertEqual([json.loads(call.arguments) for call in calls], [{}, {}])

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
