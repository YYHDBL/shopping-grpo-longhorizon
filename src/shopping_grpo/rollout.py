import json


def make_step(tool_name, parameters, action, observation, reward, done, info=None):
    return {
        "tool_name": tool_name,
        "parameters": parameters or {},
        "action": action,
        "observation": observation,
        "reward": float(reward),
        "done": bool(done),
        "info": info or {},
    }


def make_trajectory(task_id, steps, final_reward, done, instruction_text=None):
    return {
        "task_id": int(task_id),
        "instruction_text": instruction_text,
        "steps": steps,
        "final_reward": float(final_reward),
        "done": bool(done),
    }


def write_jsonl(path, trajectories):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
