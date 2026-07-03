import contextvars


CURRENT_SHOP_ENV = contextvars.ContextVar("current_shop_env", default=None)
CURRENT_SHOP_STATE = contextvars.ContextVar("current_shop_state", default=None)
CURRENT_ASSISTANT_CONTENT = contextvars.ContextVar("current_assistant_content", default=None)


def make_initial_state(task_id):
    return {
        "task_id": int(task_id),
        "total_reward": 0.0,
        "num_tool_calls": 0,
        "done": False,
        "action_history": [],
    }
