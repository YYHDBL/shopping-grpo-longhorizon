from typing import Optional
from uuid import uuid4

from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.verl_shop_context import CURRENT_SHOP_ENV, CURRENT_SHOP_STATE, make_initial_state

try:
    from verl.interactions.base import BaseInteraction
except Exception:
    class BaseInteraction:
        def __init__(self, config):
            self.config = config
            self.name = config.get("name", "shop")


def _latest_assistant_content(messages):
    for message in reversed(messages or []):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return ""


class ShopInteraction(BaseInteraction):
    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://127.0.0.1:5000")
        self.max_turns = int(config.get("max_turns", 6))
        self.env_factory = config.get("env_factory", ShopAgentEnv)
        self._instance_dict = {}

    async def start_interaction(self, instance_id: Optional[str] = None, task_id=0, **kwargs):
        instance_id = instance_id or str(uuid4())
        env = self.env_factory(base_url=self.base_url)
        try:
            initial = env.reset(int(task_id))
        except Exception:
            release = getattr(env, "release", None)
            if release is not None:
                release()
            raise
        state = make_initial_state(task_id)
        state["initial_observation"] = initial.get("instruction", "")

        CURRENT_SHOP_ENV.set(env)
        CURRENT_SHOP_STATE.set(state)
        self._instance_dict[instance_id] = {"env": env, "state": state}
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        entry = self._instance_dict[instance_id]
        env = entry["env"]
        state = entry["state"]
        CURRENT_SHOP_ENV.set(env)
        CURRENT_SHOP_STATE.set(state)

        state["last_assistant_content"] = _latest_assistant_content(messages)
        score = await self.calculate_score(instance_id)
        return True, "", score, {"reason": "assistant_final", "task_id": state["task_id"]}

    async def calculate_score(self, instance_id, **kwargs):
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return 0.0
        score = float(entry["state"].get("total_reward", 0.0))
        return max(0.0, min(1.0, score))

    async def finalize_interaction(self, instance_id, **kwargs):
        entry = self._instance_dict.pop(instance_id, None)
        if entry is not None:
            entry["env"].release()
