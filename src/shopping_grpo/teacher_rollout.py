"""使用 OpenAI 兼容工具调用采集购物 Teacher 轨迹。"""

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

from shopping_grpo.action_validation import action_guard_tool_message, action_reject_reason
from shopping_grpo.shop_http_env import ShopAgentEnv, ShopEnvironmentError, ShopHttpError
from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


SYSTEM_PROMPT = """你是一个购物 Agent，负责在 ShopSimulator 中替用户完成购买。

这是单轮购物任务：用户的完整需求只会在开头给出。不得向用户追问、确认、告别，也不要假设存在用户对话工具。你只能调用提供的标准工具与商店交互。

执行规则：
1. 每次工具返回后，先阅读最新 observation。`open_product`、`select_option` 和其他点击的值必须来自当前页面显示的可点击按钮；不要使用历史页面中的商品编号或规格。调用 `buy_now` 前，必须检查 `Buy Now 是否出现在最新 observation`；未出现时不得购买。
2. 搜索词应包含品类和用户已给出的关键属性、规格、品牌或预算线索。搜索后打开候选商品，按需查看 Description、Features、Attributes 或 Reviews 来核验。只有按钮实际出现才可调用这些子页工具；信息子页通常不能选规格或购买，若当前只显示返回按钮，立即按当前按钮返回商品详情页。
3. 选择前检查点：首次调用 `select_option` 前，先核验商品本体的用户硬约束：型号、产地、材质、功能及适用场景。任一硬约束未在当前页面证实，不得购买；继续查看或返回搜索，不能用相似商品、"通常如此"、最低价范围、促销或限购文案猜测补齐。
4. 选择和购买：只从当前页面选择用户要求的规格；同一规格组只能选择一个值。首次选中规格即结束探索阶段：这表示你已确认当前商品和规格准备购买，不得查看子页、返回搜索、再次搜索或打开其他商品。若仍有未选的必要规格，下一步只能继续 `select_option`；否则，读取选中规格后才显示的真实价格。若符合用户明确预算且 Buy Now 可见，下一步立即 `buy_now`。不得把它用于试价格或比较候选。若选中后才显示的真实价格与明确预算冲突，明确预算已冲突时，绝不为了完成任务而购买，也不得重新进入搜索或换商品。`buy_now` 是终止动作，不是跳转到结算页。
5. 轨迹有严格步数上限。不要重复无效搜索、不要调用 `think`；优先快速核验硬约束，但不要因步数不足而违背硬约束购买。
6. 不要在购买前输出最终答复、推荐总结或停止调用工具。每个未结束的 assistant 回合只输出一个工具调用；只有环境报告任务结束后才停止。
7. 若某次 tool 返回“本地动作守卫拒绝，未执行”，立即按其中当前页面允许的目标重新调用，不要重复被拒绝动作。
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
        transport=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout = timeout
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.thinking = bool(thinking)
        self.reasoning_effort = reasoning_effort
        self.transport = transport

    def complete(self, messages, tools):
        payload = {
            "model": self.model,
            "messages": messages,
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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 避免 Cloudflare 将 Python urllib 默认客户端识别为自动化流量。
            "User-Agent": "shopping-grpo-longhorizon/0.1",
        }
        url = f"{self.base_url}/chat/completions"
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
                return _response_message(response)
            except (RemoteDisconnected, TimeoutError, URLError):
                if attempt >= MODEL_COMPLETION_RETRIES:
                    raise
                time.sleep(MODEL_RETRY_DELAY_SECONDS * (attempt + 1))


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
    base_url="http://127.0.0.1:5000",
    max_steps=30,
    tools=None,
    attempt_index=0,
):
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
        "initial_result": {},
        "terminal_result": {},
        "final_reward": 0.0,
        "done": False,
        "error": None,
        "release_error": None,
    }
    env = env_factory(base_url=base_url)
    try:
        initial = env.reset(task["task_id"])
        latest_observation = initial.get("instruction", initial.get("observation", ""))
        trajectory["initial_result"] = initial
        messages = _initial_messages(task, initial)
        trajectory["messages"] = messages
        tool_schemas = tools or SHOP_TOOL_SCHEMAS
        consecutive_blocked_calls = 0

        while len(trajectory["steps"]) < int(max_steps):
            assistant = client.complete(messages, tool_schemas)
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
            step = _execute_tool_call(env, tool_call, len(trajectory["steps"]))
            trajectory["steps"].append(step)
            consecutive_blocked_calls = 0
            latest_observation = step["observation"]
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
                    "ShopSimulator infrastructure failure; collection stopped before the next task"
                )
    return written


def _is_infrastructure_failure(trajectory):
    """只在环境不可用或租约未释放时中断；普通任务失败仍保留并继续。"""
    if trajectory.get("release_error"):
        return True
    error = trajectory.get("error") or {}
    error_type = error.get("type")
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
    step.update(
        {
            "observation": result.get("instruction", result.get("observation", "")),
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


def _response_message(response):
    choice = response["choices"][0]
    message = choice["message"]
    return _plain(message)


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
    )
