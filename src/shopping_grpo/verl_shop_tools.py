import json
from uuid import uuid4

from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS, tool_call_to_action
from shopping_grpo.verl_shop_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_SHOP_ENV,
    CURRENT_SHOP_STATE,
)

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
    from verl.utils.rollout_trace import rollout_trace_op
except Exception:
    class ToolResponse:
        def __init__(self, text=""):
            self.text = text

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config = config
            self.tool_schema = tool_schema
            self.name = _schema_name(tool_schema)

    def rollout_trace_op(fn):
        return fn


def _schema_name(tool_schema):
    if isinstance(tool_schema, dict):
        return tool_schema["function"]["name"]
    return tool_schema.function.name


class ShopToolBase(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        if not getattr(self, "name", None):
            self.name = _schema_name(tool_schema)

    def get_openai_tool_schema(self):
        return self.tool_schema

    async def create(self, instance_id=None, **kwargs):
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id, parameters, **kwargs):
        env = CURRENT_SHOP_ENV.get()
        state = CURRENT_SHOP_STATE.get()
        if env is None or state is None:
            raise RuntimeError(f"Shop env/state missing for tool '{self.name}'")

        parameters = parameters or {}
        action = tool_call_to_action(self.name, parameters)
        if action is None:
            obs = parameters.get("note", "")
            inc_reward = 0.0
            done = False
            status_code = None
        else:
            result = env.step(action)
            obs = result.get("observation", "")
            inc_reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            status_code = result.get("status_code")

        state["total_reward"] += inc_reward
        state["num_tool_calls"] += 1
        if done:
            state["done"] = True
        state["action_history"].append(
            {
                "tool": self.name,
                "parameters": parameters,
                "param_str": json.dumps(parameters, sort_keys=True, ensure_ascii=False),
                "action": action,
                "inc_reward": inc_reward,
                "done": done,
                "status_code": status_code,
                "content": CURRENT_ASSISTANT_CONTENT.get() or "",
            }
        )

        return (
            ToolResponse(text=str(obs)),
            0.0,
            {"inc_reward": inc_reward, "done": done, "tool": self.name, "action": action},
        )

    async def calc_reward(self, instance_id, **kwargs):
        return 0.0

    async def release(self, instance_id, **kwargs):
        return None


SHOP_TOOL_NAMES = [schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS]


class Shop_search_products_Tool(ShopToolBase): pass
class Shop_open_product_Tool(ShopToolBase): pass
class Shop_select_option_Tool(ShopToolBase): pass
class Shop_view_description_Tool(ShopToolBase): pass
class Shop_view_features_Tool(ShopToolBase): pass
class Shop_view_reviews_Tool(ShopToolBase): pass
class Shop_view_attributes_Tool(ShopToolBase): pass
class Shop_next_page_Tool(ShopToolBase): pass
class Shop_prev_page_Tool(ShopToolBase): pass
class Shop_back_to_search_Tool(ShopToolBase): pass
class Shop_buy_now_Tool(ShopToolBase): pass
class Shop_think_Tool(ShopToolBase): pass


def get_tool_class_by_name(tool_name):
    cls_name = f"Shop_{tool_name}_Tool"
    try:
        return globals()[cls_name]
    except KeyError as exc:
        raise KeyError(f"No shopping tool class {cls_name}") from exc
