"""使用 OpenAI 兼容工具调用采集购物 Teacher 轨迹。"""

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


SYSTEM_PROMPT = """你是一个购物 Agent，负责在 ShopSimulator 中替用户完成购买。

这是单轮购物任务：用户的完整需求只会在开头给出。不得向用户追问、确认、告别，也不要假设存在用户对话工具。你只能调用提供的标准工具与商店交互。

执行规则：
1. 每次工具返回后，先阅读最新 observation。`open_product`、`select_option` 和其他点击的值必须来自当前页面显示的可点击按钮；不要使用历史页面中的商品编号或规格。调用 `buy_now` 前，必须检查 `Buy Now 是否出现在最新 observation`；未出现时不得购买。
2. 搜索词应包含品类和用户已给出的关键属性、规格、品牌或预算线索。搜索后打开候选商品，按需查看 Description、Features、Attributes 或 Reviews 来核验。这些子页面通常不能选规格或购买；查看后应使用 `prev_page` 返回商品详情页。
3. 当候选商品最符合需求时，先用 `select_option` 选择所有相关规格，再调用 `buy_now` 完成购买。若规格选项或 Buy Now 不在最新页面，先返回商品详情页。`buy_now` 是终止动作，不是跳转到结算页。
4. 轨迹有严格步数上限。不要重复无效搜索、不要调用 `think`，并在到达上限前购买当前最符合需求且已核验规格的商品；若信息仍不完美，也应选择当前最佳候选并购买。
5. 不要在购买前输出最终答复、推荐总结或停止调用工具。每个未结束的 assistant 回合只输出一个工具调用；只有环境报告任务结束后才停止。
"""


class OpenAIChatClient:
    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature=0.0,
        top_p=1.0,
        timeout=60,
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
        self.thinking = bool(thinking)
        self.reasoning_effort = reasoning_effort
        self.transport = transport

    def complete(self, messages, tools):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
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
        }
        url = f"{self.base_url}/chat/completions"
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


class ToolExecutionError(RuntimeError):
    def __init__(self, step, original):
        super().__init__(str(original))
        self.step = step
        self.original = original


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
        "initial_result": {},
        "terminal_result": {},
        "final_reward": 0.0,
        "done": False,
        "error": None,
    }
    env = env_factory(base_url=base_url)
    try:
        initial = env.reset(task["task_id"])
        trajectory["initial_result"] = initial
        messages = _initial_messages(task, initial)
        trajectory["messages"] = messages
        tool_schemas = tools or SHOP_TOOL_SCHEMAS

        while len(trajectory["steps"]) < int(max_steps):
            assistant = client.complete(messages, tool_schemas)
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                trajectory["status"] = "assistant_final"
                break
            for tool_call in tool_calls:
                if len(trajectory["steps"]) >= int(max_steps):
                    trajectory["status"] = "max_steps"
                    return trajectory
                step = _execute_tool_call(env, tool_call, len(trajectory["steps"]))
                trajectory["steps"].append(step)
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
        except Exception:
            if trajectory["error"] is None:
                trajectory["error"] = {
                    "type": "ReleaseError",
                    "message": traceback.format_exc(),
                }
                trajectory["status"] = "error"
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
    return written


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
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
