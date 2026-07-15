#!/usr/bin/env python3
import argparse
import asyncio

from shopping_grpo.verl_shop_interaction import ShopInteraction
from shopping_grpo.verl_shop_tools import Shop_search_products_Tool


class FakeEnv:
    def __init__(self, base_url=None):
        self.actions = []

    def reset(self, task_id):
        return {"observation": "fake instruction", "reward": 0.0, "done": False}

    def step(self, action):
        self.actions.append(action)
        return {
            "observation": "fake search results",
            "reward": 0.5,
            "done": False,
            "status_code": 200,
        }

    def release(self):
        return None


async def _run_dryrun(query, fake_env=True, base_url="http://127.0.0.1:5000"):
    config = {"base_url": base_url}
    if fake_env:
        config["env_factory"] = FakeEnv
    interaction = ShopInteraction(config)
    instance_id = await interaction.start_interaction("dryrun", task_id=0)
    tool = Shop_search_products_Tool({}, {"function": {"name": "search_products"}})
    response, _, info = await tool.execute(instance_id, {"query": query})
    score = await interaction.calculate_score(instance_id)
    await interaction.finalize_interaction(instance_id)
    return {
        "tool": "search_products",
        "action": info["action"],
        "response": response.text,
        "score": score,
    }


def run_fake_dryrun(query="乳胶枕"):
    return asyncio.run(_run_dryrun(query=query, fake_env=True))


def main():
    parser = argparse.ArgumentParser(description="Dry-run ShopInteraction with one search tool call.")
    parser.add_argument("--query", default="乳胶枕")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--fake-env", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(_run_dryrun(args.query, fake_env=args.fake_env, base_url=args.base_url))
    print(result)


if __name__ == "__main__":
    main()
