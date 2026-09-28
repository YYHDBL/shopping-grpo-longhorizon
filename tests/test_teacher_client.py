# Teacher 客户端协议转换单测（不触网）
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402


def make_client(response):
    return AnthropicTeacherClient(
        api_key="test",
        base_url="https://teacher.test",
        model="glm-test",
        transport=lambda url, payload, headers, timeout: response,
    )


class ConversionTest(unittest.TestCase):
    def test_round_trip_tool_call(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "should be dropped"},
                {"type": "text", "text": "先搜索"},
                {"type": "tool_use", "id": "t1", "name": "search_products",
                 "input": {"query": "乳胶枕"}},
            ],
            "stop_reason": "tool_use",
            "model": "glm-5.3-flash",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        captured = {}

        def transport(url, payload, headers, timeout):
            captured["payload"] = payload
            return response

        client = AnthropicTeacherClient(
            api_key="k", base_url="https://t.test", model="m", transport=transport)
        assistant = client.complete(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "买枕头"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "t0", "type": "function",
                        "function": {"name": "search_products",
                                     "arguments": '{"query": "x"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "t0", "content": "结果页"},
            ],
            [{
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "搜索",
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string"}}},
                },
            }],
        )
        # 请求侧：system 提取、tool 消息转 tool_result、tool_choice=any
        self.assertEqual(captured["payload"]["system"], "SYS")
        self.assertEqual(captured["payload"]["tool_choice"], {"type": "any"})
        self.assertEqual(
            captured["payload"]["tools"][0]["input_schema"]["type"], "object")
        self.assertEqual(
            captured["payload"]["messages"][2]["content"][0]["type"], "tool_result")
        # 响应侧：thinking 丢弃、text 保留、tool_calls 还原为 OpenAI 风格
        self.assertEqual(assistant["content"], "先搜索")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"],
                         "search_products")
        self.assertIn("乳胶枕", assistant["tool_calls"][0]["function"]["arguments"])

    def test_text_only_reply_has_no_tool_calls(self):
        client = make_client({"content": [{"type": "text", "text": "好的"}],
                              "usage": {}})
        assistant = client.complete(
            [{"role": "user", "content": "hi"}], [])
        self.assertEqual(assistant["content"], "好的")
        self.assertIsNone(assistant["tool_calls"])


if __name__ == "__main__":
    unittest.main()
