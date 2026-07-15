#!/usr/bin/env python3
import argparse
from pathlib import Path

from shopping_grpo.rollout import make_step, make_trajectory, write_jsonl
from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.shop_tools import tool_call_to_action


def build_mock_actions(query):
    return [{"name": "search_products", "parameters": {"query": query}}]


def run_rollout(base_url, task_id, query, output):
    steps = []
    with ShopAgentEnv(base_url=base_url) as env:
        initial = env.reset(task_id)
        for tool_call in build_mock_actions(query):
            action = tool_call_to_action(tool_call["name"], tool_call["parameters"])
            if action is None:
                continue
            result = env.step(action)
            steps.append(
                make_step(
                    tool_name=tool_call["name"],
                    parameters=tool_call["parameters"],
                    action=action,
                    observation={"text": result["instruction"]},
                    reward=result["reward"],
                    done=result["done"],
                    info={},
                )
            )

    final_reward = steps[-1]["reward"] if steps else 0.0
    done = steps[-1]["done"] if steps else False
    trajectory = make_trajectory(
        task_id=task_id,
        steps=steps,
        final_reward=final_reward,
        done=done,
        instruction_text=initial["instruction"],
    )
    write_jsonl(output, [trajectory])
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run a one-query ShopSimulator mock rollout.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--query", default="乳胶枕")
    parser.add_argument("--output", type=Path, default=Path("outputs/rollouts/stage2_mock.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    path = run_rollout(args.base_url, args.task_id, args.query, args.output)
    print(path)


if __name__ == "__main__":
    main()
