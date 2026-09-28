# Teacher 模型客户端：Anthropic Messages 协议（bigmodel 代理），适配 rollout 的 client 接口。
# 职责：OpenAI 风格 messages/tools 进，OpenAI 风格 assistant 出；thinking 块丢弃。
from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from http.client import HTTPException

RETRIES = 3
RETRY_DELAY = 2.0


class TeacherApiError(RuntimeError):
    pass


# 直连 opener：服务器对三个模型端点均可直连，代理反而依赖易断的 SSH 隧道。
from urllib.request import ProxyHandler, build_opener  # noqa: E402

_OPENER = build_opener(ProxyHandler({}))


def _default_transport(url, payload, headers, timeout):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with _OPENER.open(request, timeout=timeout) as raw:
        return json.loads(raw.read().decode("utf-8"))


class AnthropicTeacherClient:
    def __init__(
        self,
        api_key,
        base_url,
        model,
        temperature=1.0,
        top_p=0.95,
        max_tokens=2048,
        timeout=120,
        transport=None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.transport = transport or _default_transport
        self.last_usage = None
        self.total_usage = {"input": 0, "output": 0, "cached": 0}

    def _to_anthropic(self, messages, tools):
        system_parts = []
        converted = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(message.get("content") or "")
                continue
            if role == "user":
                converted.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message.get("content") or ""}],
                })
                continue
            if role == "assistant":
                blocks = []
                text = message.get("content")
                if isinstance(text, str) and text.strip():
                    blocks.append({"type": "text", "text": text})
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                    except (TypeError, json.JSONDecodeError):
                        arguments = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "input": arguments,
                    })
                converted.append({"role": "assistant", "content": blocks or [
                    {"type": "text", "text": ""}
                ]})
                continue
            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id"),
                        "content": message.get("content") or "",
                    }],
                })
        anthropic_tools = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"].get("parameters")
                or {"type": "object", "properties": {}},
            }
            for tool in tools
        ]
        return "\n\n".join(part for part in system_parts if part), converted, anthropic_tools

    def complete(self, messages, tools):
        system, converted, anthropic_tools = self._to_anthropic(messages, tools)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": converted,
        }
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            payload["tool_choice"] = {"type": "any"}
        if system:
            payload["system"] = system
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        started = time.monotonic()
        last_error = None
        for attempt in range(RETRIES + 1):
            try:
                response = self.transport(
                    f"{self.base_url}/v1/messages", payload, headers, self.timeout
                )
                self.last_usage = response.get("usage") or {}
                usage = self.last_usage
                self.total_usage["input"] += usage.get("input_tokens") or 0
                self.total_usage["output"] += usage.get("output_tokens") or 0
                self.total_usage["cached"] += usage.get("cache_read_input_tokens") or 0
                self.last_latency_ms = round((time.monotonic() - started) * 1000, 1)
                return self._to_openai_assistant(response)
            except HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                last_error = f"HTTP {exc.code}: {body}"
                if exc.code != 429 and exc.code < 500:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError, HTTPException, OSError) as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * (2**attempt))
        raise TeacherApiError(f"teacher request failed: {last_error}")

    def _to_openai_assistant(self, response):
        text_parts = []
        tool_calls = []
        for block in response.get("content") or []:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text") or "")
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(
                            block.get("input") or {}, ensure_ascii=False
                        ),
                    },
                })
        return {
            "role": "assistant",
            "content": "\n".join(part for part in text_parts if part) or None,
            "tool_calls": tool_calls or None,
            "stop_reason": response.get("stop_reason"),
            "model_reported": response.get("model"),
        }
