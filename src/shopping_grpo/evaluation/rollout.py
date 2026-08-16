"""使用 OpenAI-compatible 工具调用运行 Shopping Agent 评测轨迹。

这里负责把模型回复、工具守卫和 ShopSimulator 串成一条可回放记录；它是评测采集
入口，不负责修改环境仓库或训练模型。
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from urllib.error import URLError
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from shopping_grpo.environment.actions import action_guard_tool_message, action_reject_reason
from shopping_grpo.environment.context import (
    ContextBudgetError,
    VllmChatTokenCounter,
    VllmTextTokenCounter,
    compact_chat_messages,
)
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.environment.client import ShopAgentEnv, ShopEnvironmentError, ShopHttpError
from shopping_grpo.environment.tools import (
    SHOP_TOOL_SCHEMAS,
    tool_call_to_action,
)
from shopping_grpo.environment.observation import render_structured_observation


SYSTEM_PROMPT = """你是一个购物 Agent，负责在 ShopSimulator 中替用户完成一次单轮购物任务。

用户的完整需求只会在开头给出。不得向用户追问、确认、告别，也不要假设存在用户对话工具。你只能调用提供的标准工具与商店交互。目标是在有限步骤内找到整体最符合需求的可购买商品；精确满足全部要求的商品最好，经过有效探索仍无法完成时应合理结束，不能错误购买或无效循环。

执行规则：
1. 动作合法性与历史比较。当前页面是动作合法性的唯一依据；历史 observation 可以用于记住和比较候选，但不能直接点击历史页面中的 ASIN、按钮或规格。每次工具返回后先阅读最新 observation 中的“可点击的按钮”，每个 assistant 回合只调用一个工具。
2. 页面状态与参数。Description、Features、Reviews、Attributes 都是信息子页：一旦进入这类子页，必须先调用当前页面可见的 `prev_page` 或 `back_to_search` 返回；不得直接切换到另一个信息子页、选择规格、购买或搜索。无参数工具（查看、翻页、返回、购买）必须传严格的 `{}`；只有 `search_products` 使用 `query`、`open_product` 使用 `asin`、`select_option` 使用 `value`。
3. 搜索与候选探索。查询应简洁，优先使用品类和最有区分度的品牌、型号、核心功能或规格，不要机械复制整段需求。结果不理想时，缩短查询、更换真正不同的关键词或翻页；出现有希望的商品时应打开核验。不要重复相同查询，也不要只做同义改写却反复得到相同候选。
4. 固定选择优先级。按“品类 > 预算 > 品牌 > 型号与核心功能 > 规格属性”比较候选。品类必须正确；选择具体规格后的实际价格不得超过用户明确预算。品类不符、价格未知或超预算时绝不能购买。在通过这两个门槛的候选中，依次优先满足品牌、型号与核心功能、规格属性；全部要求都满足的候选最好。
5. 证据、规格与购买。品牌、型号、规格、功能和价格优先依据结构化字段、商品详情、Description、Features 和 Attributes；Reviews 只用于辅助判断使用体验，不能用于确认型号、官方功能、规格或价格。先完成必要的商品核验，再为最终候选补齐当前商品所有影响可购买 variant 的必要规格轴；不要为了临时浏览而随意选择规格。同一规格轴只选择一个当前页面可见的值，并以完整 variant 的实际价格判断预算。只有 `Buy Now` 当前可见，且商品通过品类和预算门槛，并在其余维度上是已核验候选中的最佳选择时，才调用 `buy_now`。
6. 主动结束。经过多次有实质差异的搜索和多个候选核验，仍没有可接受商品，并且当前没有明显值得继续核验的候选时，调用 `finish_without_purchase`。不得过早结束，也不要为了增加搜索次数继续无效探索；是否达到结束资格由环境判断。
7. 防止循环和非法动作。不要连续重复同一动作，也不要在相同结果、商品或子页之间无目的往返；后续操作应带来新候选、新商品信息、新规格选择或新的需求证据。不要调用 `think` 工具。若 tool 返回“本地动作守卫拒绝，未执行”，依据错误消息和最新 observation 改为一个合法动作，不要重复被拒绝的调用。不要在任务结束前输出最终答复或推荐总结；只有环境报告任务结束后才停止。
"""


MAX_BLOCKED_TOOL_CALLS = 3
MODEL_COMPLETION_RETRIES = 2
MODEL_RETRY_DELAY_SECONDS = 1


def rollout_interrupted(signum, frame):
    """将终止信号转为 KeyboardInterrupt，使 collect_for_task 的 finally 释放环境。"""
    raise KeyboardInterrupt


class OpenAIChatClient:
    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature=0.0,
        top_p=1.0,
        timeout=60,
        max_tokens=512,
        thinking=False,
        reasoning_effort="high",
        context_window=None,
        context_safety_margin=512,
        context_compaction_enable=False,
        observation_token_budget=None,
        observation_detail_token_budget=4096,
        observation_generic_token_budget=768,
        observation_search_top_k=20,
        token_counter=None,
        observation_token_counter=None,
        transport=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.responses_api = self.base_url.endswith("/responses")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout = timeout
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.thinking = bool(thinking)
        self.reasoning_effort = reasoning_effort
        self.context_window = int(context_window) if context_window else None
        self.context_safety_margin = int(context_safety_margin)
        self.context_compaction_enable = bool(context_compaction_enable)
        self.observation_token_budget = (
            int(observation_token_budget) if observation_token_budget else None
        )
        self.observation_search_top_k = int(observation_search_top_k)
        self.observation_detail_token_budget = int(observation_detail_token_budget)
        self.observation_generic_token_budget = int(observation_generic_token_budget)
        if self.context_window is not None:
            if self.context_window <= self.max_tokens + self.context_safety_margin:
                raise ValueError("context_window must exceed max_tokens plus context_safety_margin")
            self.token_counter = token_counter or VllmChatTokenCounter(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        else:
            self.token_counter = token_counter
        if self.observation_token_budget is not None:
            if self.observation_token_budget < 64:
                raise ValueError("observation_token_budget must be at least 64")
            self.observation_token_counter = (
                observation_token_counter
                or VllmTextTokenCounter(
                    model=self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            )
        else:
            self.observation_token_counter = observation_token_counter
        self.last_context_event = None
        self.last_context_tokens = None
        self.transport = transport

    def complete(self, messages, tools):
        """请求模型下一轮回复，并在上下文超限时按配置压缩历史。"""
        self.last_context_event = None
        self.last_context_tokens = None
        request_messages = messages
        if self.context_window is not None:
            input_budget = self.context_window - self.max_tokens - self.context_safety_margin
            original_tokens = int(self.token_counter(messages, tools))
            self.last_context_tokens = original_tokens
            if original_tokens > input_budget:
                if not self.context_compaction_enable:
                    raise ContextBudgetError(
                        f"prompt uses {original_tokens} tokens, above input budget {input_budget}"
                    )
                request_messages, stats = compact_chat_messages(
                    messages,
                    tools,
                    count_tokens=self.token_counter,
                    max_input_tokens=input_budget,
                )
                if stats.removed_groups:
                    self.last_context_event = stats.to_dict()
        if self.responses_api:
            payload = {
                "model": self.model,
                "input": _responses_input(request_messages),
                "tools": _responses_tools(tools),
                "tool_choice": "auto",
                "max_output_tokens": self.max_tokens,
            }
        else:
            payload = {
                "model": self.model,
                "messages": request_messages,
                "tools": tools,
                "tool_choice": "auto",
                # 约束单个 assistant 回合的输出；--max-model-len 只限制上下文，
                # 不能防止模型在未调用工具时持续生成纯文本。
                "max_tokens": self.max_tokens,
            }
        if self.thinking:
            # DeepSeek tool-call thinking requires reasoning_content in later messages.
            payload.update(
                {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": self.reasoning_effort,
                }
            )
        else:
            payload.update({"temperature": self.temperature, "top_p": self.top_p})
            # OpenCode Go defaults DeepSeek V4 to thinking enabled. Omitting the
            # field therefore does not mean disabled; be explicit for this model
            # family without sending a provider-specific field to local vLLM.
            if self.model.casefold().startswith("deepseek-v4"):
                payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 避免 Cloudflare 将 Python urllib 默认客户端识别为自动化流量。
            "User-Agent": "shopping-grpo-longhorizon/0.1",
        }
        url = self.base_url if self.responses_api else f"{self.base_url}/chat/completions"
        for attempt in range(MODEL_COMPLETION_RETRIES + 1):
            try:
                if self.transport is not None:
                    response = self.transport(url, payload, headers, self.timeout)
                else:
                    request = Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout) as raw:
                        response = json.loads(raw.read().decode("utf-8"))
                return _response_message(response, responses_api=self.responses_api)
            except (RemoteDisconnected, TimeoutError, URLError):
                if attempt >= MODEL_COMPLETION_RETRIES:
                    raise
                time.sleep(MODEL_RETRY_DELAY_SECONDS * (attempt + 1))

    def project_observation(self, tool_name, observation, parameters=None):
        if self.observation_token_budget is None:
            return str(observation), None
        visible, meta = project_observation(
            tool_name=tool_name,
            observation=observation,
            parameters=parameters,
            count_tokens=self.observation_token_counter,
            token_budget=self.observation_token_budget,
            detail_token_budget=self.observation_detail_token_budget,
            generic_token_budget=self.observation_generic_token_budget,
            search_top_k=self.observation_search_top_k,
        )
        return visible, meta.to_dict()


class ToolExecutionError(RuntimeError):
    def __init__(self, step, original):
        super().__init__(str(original))
        self.step = step
        self.original = original


class CollectionInfrastructureError(RuntimeError):
    """环境租约未恢复时，阻止采集器继续污染后续任务。"""


def load_tasks(path):
    tasks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = _task_id(row)
            if task_id is None:
                raise ValueError("task row is missing task_id")
            row = dict(row)
            row["task_id"] = int(task_id)
            tasks.append(row)
    return tasks


def completed_task_attempts(path):
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task_id" in row:
                done.add((int(row["task_id"]), int(row.get("attempt_index", 0))))
    return done


def collect_for_task(
    task,
    client,
    env_factory=ShopAgentEnv,
    base_url="http://127.0.0.1:5700",
    max_steps=30,
    tools=None,
    attempt_index=0,
):
    """执行一个任务并返回完整轨迹；所有异常都会被写入轨迹后再释放环境。"""
    trajectory = {
        "trajectory_id": str(uuid4()),
        "task_id": int(task["task_id"]),
        "attempt_index": int(attempt_index),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "messages": [],
        "steps": [],
        "blocked_tool_calls": [],
        "tool_call_truncations": [],
        "context_compactions": [],
        "context_turn_tokens": [],
        "initial_result": {},
        "terminal_result": {},
        "final_reward": 0.0,
        "done": False,
        "error": None,
        "release_error": None,
    }
    env = env_factory(base_url=base_url)
    try:
        # reset 建立任务状态；后续每一轮只允许一个工具调用。
        initial = env.reset(task["task_id"])
        if initial.get("observation_state") is not None:
            latest_observation = render_structured_observation(
                initial["observation_state"]
            )
        else:
            latest_observation = initial.get(
                "instruction", initial.get("observation", "")
            )
        trajectory["initial_result"] = initial
        messages = _initial_messages(task, initial)
        trajectory["messages"] = messages
        tool_schemas = tools or SHOP_TOOL_SCHEMAS
        consecutive_blocked_calls = 0
        latest_observation_truncated = False

        while len(trajectory["steps"]) < int(max_steps):
            # 先请求模型，再校验动作；工具结果会追加到 messages，成为下一轮上下文。
            assistant = client.complete(messages, tool_schemas)
            context_tokens = getattr(client, "last_context_tokens", None)
            if context_tokens is not None:
                trajectory["context_turn_tokens"].append(
                    {
                        "step_index": len(trajectory["steps"]),
                        "input_tokens": int(context_tokens),
                    }
                )
            context_event = getattr(client, "last_context_event", None)
            if context_event:
                trajectory["context_compactions"].append(
                    {
                        "step_index": len(trajectory["steps"]),
                        **context_event,
                    }
                )
            assistant, dropped_tool_calls = _enforce_serial_tool_call(assistant)
            if dropped_tool_calls:
                trajectory["tool_call_truncations"].append(
                    {
                        "message_index": len(messages),
                        "kept_tool_call_id": assistant["tool_calls"][0].get("id"),
                        "dropped_tool_calls": dropped_tool_calls,
                    }
                )
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                messages.append(assistant)
                trajectory["status"] = "assistant_final"
                break
            tool_call = tool_calls[0]
            try:
                name, arguments = _tool_call_name_args(tool_call)
                reason = action_reject_reason(
                    name,
                    arguments,
                    latest_observation,
                )
            except Exception as exc:
                reason = f"invalid_tool_call:{exc.__class__.__name__}"
            if reason:
                consecutive_blocked_calls += 1
                trajectory["blocked_tool_calls"].append(
                    {
                        "step_index": len(trajectory["steps"]),
                        "tool_call": tool_call,
                        "reason": reason,
                        "consecutive_count": consecutive_blocked_calls,
                        "latest_observation_truncated": latest_observation_truncated,
                    }
                )
                messages.append(assistant)
                messages.append(action_guard_tool_message(tool_call, reason, latest_observation))
                if consecutive_blocked_calls >= MAX_BLOCKED_TOOL_CALLS:
                    trajectory["status"] = "invalid_action_limit"
                    break
                continue
            if len(trajectory["steps"]) >= int(max_steps):
                trajectory["status"] = "max_steps"
                return trajectory
            messages.append(assistant)
            # 只有通过当前 observation 守卫的调用才会触碰环境并消耗一个执行步骤。
            step = _execute_tool_call(env, tool_call, len(trajectory["steps"]))
            raw_observation = step["observation"]
            projector = getattr(client, "project_observation", None)
            if projector is not None:
                visible_observation, projection = projector(
                    step["tool_name"],
                    raw_observation,
                    step["parameters"],
                )
                if projection is not None:
                    step["raw_observation"] = raw_observation
                    step["observation"] = visible_observation
                    step["projection"] = projection
            trajectory["steps"].append(step)
            consecutive_blocked_calls = 0
            latest_observation = step["observation"]
            latest_observation_truncated = bool(
                (step.get("projection") or {}).get("truncated")
            )
            messages.append(_tool_message(tool_call, step))
            if step["done"]:
                trajectory["status"] = "done"
                trajectory["terminal_result"] = step["result"]
                trajectory["final_reward"] = step["reward"]
                trajectory["done"] = True
                return trajectory
        else:
            trajectory["status"] = "max_steps"
        if trajectory["steps"]:
            trajectory["final_reward"] = trajectory["steps"][-1]["reward"]
    except ToolExecutionError as exc:
        trajectory["steps"].append(exc.step)
        trajectory["status"] = "error"
        trajectory["error"] = {
            "type": exc.original.__class__.__name__,
            "message": str(exc.original),
            "traceback": "".join(traceback.format_exception(exc.original)),
        }
    except Exception as exc:
        trajectory["status"] = "error"
        trajectory["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            env.release()
        except Exception as exc:
            trajectory["release_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            trajectory["status"] = "environment_release_failed"
    return trajectory


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_tasks(
    tasks,
    client,
    output_path,
    base_url,
    max_steps=30,
    env_factory=ShopAgentEnv,
    attempts_per_task=1,
):
    attempts_per_task = int(attempts_per_task)
    if attempts_per_task < 1:
        raise ValueError("attempts_per_task must be at least 1")
    done = completed_task_attempts(output_path)
    written = []
    for task in tasks:
        task_id = int(task["task_id"])
        for attempt_index in range(attempts_per_task):
            if (task_id, attempt_index) in done:
                continue
            trajectory = collect_for_task(
                task,
                client=client,
                env_factory=env_factory,
                base_url=base_url,
                max_steps=max_steps,
                attempt_index=attempt_index,
            )
            append_jsonl(output_path, [trajectory])
            written.append(trajectory)
            if _is_infrastructure_failure(trajectory):
                raise CollectionInfrastructureError(
                    "collection infrastructure failure; stopped before the next task"
                )
    return written


def _is_infrastructure_failure(trajectory):
    """环境或模型服务不可用时中断；普通任务失败仍保留并继续。"""
    if trajectory.get("release_error"):
        return True
    error = trajectory.get("error") or {}
    error_type = error.get("type")
    if error_type in {URLError.__name__, RemoteDisconnected.__name__, TimeoutError.__name__}:
        return True
    if error_type == ShopHttpError.__name__:
        return True
    return (
        error_type == ShopEnvironmentError.__name__
        and "Unable to get available environment resource" in error.get("message", "")
    )


def _execute_tool_call(env, tool_call, step_index):
    name, arguments = _tool_call_name_args(tool_call)
    action = tool_call_to_action(name, arguments)
    result = {"instruction": arguments.get("note", ""), "reward": 0.0, "done": False}
    step = {
        "step_index": step_index,
        "tool_call": tool_call,
        "tool_name": name,
        "parameters": arguments,
        "env_action": action,
        "observation": "",
        "reward": 0.0,
        "done": False,
        "result": {},
    }
    if action is not None:
        try:
            result = env.step(action)
        except Exception as exc:
            step["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            raise ToolExecutionError(step, exc) from exc
    if result.get("observation_state") is not None:
        observation = render_structured_observation(result["observation_state"])
    else:
        observation = result.get("instruction", result.get("observation", ""))
    step.update(
        {
            "observation": observation,
            "reward": float(result.get("reward", 0.0)),
            "done": bool(result.get("done", False)),
            "result": result,
        }
    )
    return step


def _tool_message(tool_call, step):
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": step["tool_name"],
        "content": step["observation"],
    }


def _enforce_serial_tool_call(assistant):
    """每轮只把一个工具调用交给环境，防止在旧 observation 上批量点击。"""
    tool_calls = assistant.get("tool_calls") or []
    if len(tool_calls) <= 1:
        return assistant, []
    serial_assistant = dict(assistant)
    serial_assistant["tool_calls"] = [tool_calls[0]]
    return serial_assistant, list(tool_calls[1:])


def _initial_messages(task, initial):
    prompt = task.get("prompt")
    if prompt:
        messages = [dict(message) for message in prompt]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": initial.get("instruction", "")})
    return messages


def _task_id(row):
    if "task_id" in row:
        return row["task_id"]
    extra = row.get("extra_info") or {}
    if "task_id" in extra:
        return extra["task_id"]
    kwargs = extra.get("interaction_kwargs") or {}
    return kwargs.get("task_id")


def _tool_call_name_args(tool_call):
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_args = function.get("arguments") or "{}"
    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    return name, arguments


def _response_message(response, responses_api=False):
    if responses_api:
        return _responses_message(response)
    choice = response["choices"][0]
    message = choice["message"]
    return _plain(message)


def _responses_tools(tools):
    """将 Chat Completions function schema 转为 Responses function schema。"""
    converted = []
    for tool in tools or []:
        function = tool.get("function") or {}
        converted.append(
            {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters") or {"type": "object"},
                "strict": True,
            }
        )
    return converted


def _responses_input(messages):
    """将内部 Chat 消息转换为 Responses input items。"""
    converted = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": str(message.get("content", "")),
                }
            )
            continue
        tool_calls = message.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                converted.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            continue
        if role in {"system", "user", "assistant"}:
            converted.append(
                {
                    "role": role,
                    "content": str(message.get("content") or ""),
                }
            )
    return converted


def _responses_message(response):
    """将 Responses output items 还原为现有 rollout 循环使用的 assistant 消息。"""
    tool_calls = []
    text_parts = []
    for item in response.get("output") or []:
        if item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
        elif item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    text_parts.append(content.get("text", ""))
    message = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    return value


def client_from_env(
    model=None,
    base_url=None,
    api_key=None,
    temperature=0.0,
    top_p=1.0,
    timeout=60,
    max_tokens=512,
    thinking=False,
    reasoning_effort="high",
    context_window=None,
    context_safety_margin=512,
    context_compaction_enable=False,
    observation_token_budget=None,
    observation_detail_token_budget=4096,
    observation_generic_token_budget=768,
    observation_search_top_k=20,
):
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("api_key or OPENAI_API_KEY is required")
    return OpenAIChatClient(
        model=model or os.environ.get("OPENAI_MODEL", "deepseek-chat"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        max_tokens=max_tokens,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        context_window=context_window,
        context_safety_margin=context_safety_margin,
        context_compaction_enable=context_compaction_enable,
        observation_token_budget=observation_token_budget,
        observation_detail_token_budget=observation_detail_token_budget,
        observation_generic_token_budget=observation_generic_token_budget,
        observation_search_top_k=observation_search_top_k,
    )
