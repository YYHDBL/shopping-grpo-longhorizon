import json
import logging
from typing import Dict, Any, Optional

# Constants
MAX_HISTORY_LENGTH = 42
LOG_FILE = "shop_agent.log"

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def _handle_reset_action(
    env: Any,
    env_idx: int,
    task_idx: Optional[int],
    include_trace_target: bool = False,
) -> Dict[str, Any]:
    """
    Handle reset action

    Args:
        env: Environment object
        env_idx: Environment index
        task_idx: Task index

    Returns:
        Dictionary containing reset information
    """
    logger.info(f"[Reset] Starting task {task_idx}, environment index: {env_idx}")
    message = f"Task {task_idx} started"
    env.reset(idx=task_idx)
    return_info = {
        'instruction': env.instruction_text,
        'instruction_simple': env.instruction_simple,
        'goal_options': env.goal_options,
        'message': message,
        'env_idx': env_idx,
        'idx': task_idx,
        'environment_version': getattr(
            env.server,
            "environment_version",
            "shopsimulator-environment-v2.1",
        ),
        "observation_state": env.structured_observation(),
    }
    if include_trace_target:
        goal = env.server.goals[task_idx]
        return_info["_trace_target"] = {
            "asin": goal["asin"],
            "options": goal["goal_options"],
        }
    if hasattr(env, 'user_persona') and env.user_persona is not None:
        return_info['user_persona'] = env.user_persona
        return_info['reason_key'] = env.reason_key
    return return_info


def _extract_action_from_response(response: str) -> str:
    """
    Extract action from response

    Args:
        response: Raw response string

    Returns:
        Extracted action string
    """
    if "\nAction: " in response:
        return response.split("\nAction: ")[1]
    return response


def _format_available_actions(available_actions: Dict[str, Any]) -> str:
    """
    Format available actions information

    Args:
        available_actions: Dictionary of available actions

    Returns:
        Formatted action text
    """
    clickables = available_actions['clickables'].copy()
    if "search" in clickables:
        clickables.remove("search")

    return (
        f"\n\n搜索功能是否可用: {available_actions['has_search_bar']}"
        f"\n\n可点击的按钮: {json.dumps(clickables, ensure_ascii=False)}"
    )


def _handle_interact_action(
    env: Any,
    env_idx: int,
    response: str
) -> Dict[str, Any]:
    """
    Handle interact action

    Args:
        env: Environment object
        env_idx: Environment index
        response: Response string

    Returns:
        Dictionary containing interaction information
    """
    logger.info(
        f"[Interact] Received response: {response}, "
        f"environment index: {env_idx}, Session ID: {env.session}"
    )

    # Normalize response format
    normalized_response = response.replace("\\n", "\n")
    env.history.append({'role': 'assistant', 'content': normalized_response})

    # Extract action
    action_str = _extract_action_from_response(normalized_response)

    # Execute environment step
    observation, status, info = env.step(action_str)
    done, reward = status['done'], status['reward']

    # Get and format available actions
    available_actions = env.get_available_actions()
    action_text = _format_available_actions(available_actions)
    observation = observation + action_text

    # Extract status information
    if done:
        reward_detail = status["reward_detail"]
        purchase = status['purchase']
        goal = status['goal']
    else:
        reward_detail = {}
        purchase = {}
        goal = {}

    # Build return information
    return_info = {
        "done": done,
        "reward": reward,
        "instruction": observation,
        "message": "Continue interaction",
        "env_idx": env_idx,
        "idx": env.session,
        "reward_detail": reward_detail,
        "purchase": purchase,
        "goal": goal,
        "over": len(env.history) > MAX_HISTORY_LENGTH or done,
        "observation_state": env.structured_observation(),
    }
    if done:
        return_info["termination_reason"] = status.get(
            "termination_reason", "environment_done"
        )
        return_info["reward_valid"] = bool(status.get("reward_valid", True))
    if status.get("progress") is not None:
        return_info["progress"] = status["progress"]

    return return_info


def shop_agent(
    env: Any,
    env_idx: int,
    action: str,
    idx: Optional[int] = None,
    response: Optional[str] = None,
    include_trace_target: bool = False,
) -> Dict[str, Any]:
    """
    Main shop agent function that handles environment reset and interaction actions

    Args:
        env: Environment object
        env_idx: Environment index
        action: Action type ("reset" or "interact")
        idx: Task index, only used for reset action
        response: Response string, only used for interact action

    Returns:
        Dictionary containing action processing results

    Raises:
        ValueError: When action is not "reset" or "interact"
    """
    if action == "reset":
        if idx is None:
            raise ValueError("reset action requires idx parameter")
        return _handle_reset_action(
            env,
            env_idx,
            idx,
            include_trace_target=include_trace_target,
        )

    elif action == "interact":
        if response is None:
            raise ValueError("interact action requires response parameter")
        return _handle_interact_action(env, env_idx, response)

    else:
        raise ValueError(f"Unknown action type: {action}, supported actions: 'reset', 'interact'")
