"""veRL parser for Qwen3/Qwen3.5 Coder-style XML tool calls."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

try:
    from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
except ImportError:  # pragma: no cover - lightweight local tests
    @dataclass
    class FunctionCall:
        name: str
        arguments: str

    class ToolParser:
        _registry = {}

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        @classmethod
        def register(cls, name):
            def decorator(subclass):
                cls._registry[name] = subclass
                return subclass

            return decorator


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


@ToolParser.register("qwen3_coder")
class Qwen3CoderToolParser(ToolParser):
    """Parse the XML grammar emitted by Qwen3-Coder and Qwen3.5 chat templates."""

    async def extract_tool_calls(self, responses_ids: list[int], tools=None):
        del tools
        text = await asyncio.to_thread(self.tokenizer.decode, responses_ids)
        calls = []
        for match in TOOL_CALL_RE.finditer(text):
            parameters = {
                name: _trim_wrapping_newline(value)
                for name, value in PARAMETER_RE.findall(match.group(2))
            }
            calls.append(
                FunctionCall(
                    name=match.group(1),
                    arguments=json.dumps(parameters, ensure_ascii=False),
                )
            )
        return TOOL_CALL_RE.sub("", text).strip(), calls


def _trim_wrapping_newline(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value
