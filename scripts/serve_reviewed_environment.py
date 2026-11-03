"""Serve explicit-tool observations and optional reviewed scoring through the existing ShopSimulator routes."""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP = ROOT / "environments/ShopSimulator/shop_env"
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5700)
parser.add_argument("--slots", type=int, default=2)
parser.add_argument(
    "--products", type=Path, help="Explicit product-data snapshot for an isolated experiment"
)
parser.add_argument(
    "--reviewed-reward",
    action="store_true",
    help="Isolated reviewed Reward v3 policy, recorded in reward evidence",
)
parser.add_argument(
    "--option-evidence",
    type=Path,
    help="Versioned public option-image evidence manifest; no target answers",
)
args = parser.parse_args()
if args.option_evidence:
    if not args.option_evidence.is_file():
        parser.error("Public option evidence manifest missing")
    os.environ["SHOP_PUBLIC_OPTION_EVIDENCE"] = str(args.option_evidence.resolve())
os.environ["SHOPSIM_PUBLIC_SUPPORT"] = "0"
if args.products and not args.products.is_file():
    parser.error("Product-data snapshot does not exist")
os.environ["SHOPSIM_ENV_SLOTS"] = str(args.slots)
sys.path[:0] = [str(SHOP), str(SHOP / "shop_env")]
import pack_api

if args.reviewed_reward:
    from web_agent_site.engine import reviewed_reward
    from web_agent_site.envs import web_agent_text_env

    web_agent_text_env.evaluate_purchase = reviewed_reward.evaluate_purchase
    web_agent_text_env.evaluate_candidate_eligibility = (
        reviewed_reward.evaluate_candidate_eligibility
    )

if args.products:
    pack_api.WebAgentTextEnv = partial(
        pack_api.WebAgentTextEnv, file_path=str(args.products.resolve())
    )

if __name__ == "__main__":
    pack_api.initialize_environments()
    pack_api.app.run(host="127.0.0.1", port=args.port)
