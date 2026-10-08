"""Compose independent A–G evaluation switches; never start a service or model."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def verify_catalog_pair(products, index):
    digest = hashlib.sha256()
    with products.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT payload FROM manifest").fetchone()
        manifest = json.loads(row[0])
    if manifest.get("product_data_sha256") != digest.hexdigest():
        raise ValueError("Catalog and search index SHA-256 do not match")
    return digest.hexdigest()


def compose(disabled, products, index, output):
    disabled = set(disabled)
    if disabled - set("ABCDEFG"):
        raise ValueError("Unknown component")
    products, index, output = (Path(p).resolve() for p in (products, index, output))
    digest = verify_catalog_pair(products, index)
    agent = yaml.safe_load((ROOT / "configs/agent_loop.yaml").read_text())
    config = agent[0]
    config["_target_"] = (
        "shopping_grpo.training.grpo.adapter.enhanced_agent.EnhancedShoppingAgentLoop"
    )
    config["select_original_reward"] = True
    if "A" not in disabled:
        config["system_prompt_file"] = str(ROOT / "configs/components/system_prompt.txt")
    for component, filename in (("D", "observation.json"), ("E", "context.json")):
        if component not in disabled:
            config.update(json.loads((ROOT / "configs/components" / filename).read_text()))
    tools = json.loads((ROOT / "configs/tools.json").read_text())
    for tool in tools["tools"]:
        tool["class_name"] = "shopping_grpo.training.grpo.adapter.enhanced_tools.ShopSimulatorTool"
        tool["config"].update(
            purchase_checks="B" not in disabled, interface_support="C" not in disabled
        )
    service = {
        "argv": [
            "python",
            str(ROOT / "scripts/serve_reviewed_environment.py"),
            "--products",
            str(products),
        ],
        "env": {"SHOP_SEARCH_INDEX": str(index), "SHOPSIM_PUBLIC_SUPPORT": "0"},
    }
    if "G" not in disabled:
        service["argv"].append("--reviewed-reward")
    protocol = {
        "components": {c: c not in disabled for c in "ABCDEFG"},
        "catalog_role": "original" if "F" in disabled else "reviewed",
        "product_sha256": digest,
        "products": str(products),
        "index": str(index),
        "automatic_details": False,
        "original_reward": True,
        "search_serialized": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (
        ("agent_loop.yaml", agent),
        ("tools.json", tools),
        ("service.json", service),
        ("protocol.json", protocol),
    ):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return agent, tools, service, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable", nargs="*", choices=list("ABCDEFG"), default=[])
    parser.add_argument(
        "--products",
        type=Path,
        required=True,
        help="Reviewed catalog with F enabled; original catalog with F disabled",
    )
    parser.add_argument(
        "--index", type=Path, required=True, help="Index built from that exact catalog"
    )
    parser.add_argument("--output", type=Path, required=True, help="New configuration directory")
    args = parser.parse_args()
    compose(args.disable, args.products, args.index, args.output)


if __name__ == "__main__":
    main()
