import hashlib
import gzip
import json
import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV_ROOT = ROOT / "environments/ShopSimulator/shop_env"
sys.path.append(str(SHOP_ENV_ROOT))

from web_agent_site.engine.constraints import explicit_budget_from_instruction
from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.variant_price import (
    candidate_options_for_evaluation,
    resolve_variant_price,
)


PRICE_HINT = re.compile(
    r"(?:预算|价格|售价|价钱|价位|总价|费用|成本|花费|多少钱|"
    r"[零一二三四五六七八九十百千万两\d]+(?:\.\d+)?\s*(?:元|块|钱)|"
    r"\d+(?:\.\d+)?\s*(?:左右|上下|出头))"
)
EXCLUDED_SOURCE_TASK_IDS = {
    1212, 1520, 2352, 3025, 3038, 3247, 3368, 3797, 4071, 4169,
    4206, 4364, 4604, 4786, 4925, 5377, 5475, 5510, 5633, 5703,
    5831, 5904, 6143, 6520, 6536, 6641, 7058, 7729, 8583, 9214,
    10099, 10197, 10270, 10463, 10633, 11157, 11168, 11361, 11699,
    11752, 11773, 12065, 12162, 12738, 12759, 12860, 13231, 13572,
    13968, 14040, 15082, 15567, 16367, 16816, 16830, 17352, 17393,
    17417, 17809, 18036, 18292, 18381, 18617, 18717, 18749, 19180, 19400,
    19479, 20117, 20133, 20345, 20967, 21044, 21311, 21402, 21519,
    21684, 21785, 21988, 22029, 22340,
}


class EvaluationDatasetTests(unittest.TestCase):
    def test_price_hint_detects_numeric_price_without_keyword(self):
        self.assertIsNotNone(PRICE_HINT.search("控制在15元以内"))

    def test_final_200_clean_manifest_and_blind_guard_match(self):
        tasks_path = ROOT / "data/evaluation/tasks.jsonl"
        task_bytes = tasks_path.read_bytes()
        task_ids = [json.loads(line)["task_id"] for line in task_bytes.decode().splitlines()]
        metadata = json.loads((ROOT / "data/evaluation/metadata.json").read_text())
        guarded = json.loads(
            (ROOT / "src/shopping_grpo/resources/blind_final_task_ids.json").read_text()
        )
        guard = json.loads(
            (ROOT / "src/shopping_grpo/resources/blind_guard.json").read_text()
        )

        self.assertEqual(len(task_ids), 200)
        self.assertEqual(len(set(task_ids)), 200)
        self.assertFalse(EXCLUDED_SOURCE_TASK_IDS.intersection(task_ids))
        self.assertEqual(metadata["name"], "Final-200-Clean")
        self.assertEqual(metadata["tasks"], 200)
        self.assertEqual(
            set(metadata["curation"]["excluded_source_task_ids"]),
            EXCLUDED_SOURCE_TASK_IDS,
        )
        self.assertEqual(len(metadata["curation"]["replacement_task_ids"]), 81)
        self.assertTrue(
            set(metadata["curation"]["replacement_task_ids"]).issubset(task_ids)
        )
        self.assertEqual(metadata["sha256"], hashlib.sha256(task_bytes).hexdigest())
        self.assertEqual(guarded["task_ids"], task_ids)
        self.assertEqual(guard["task_count"], 200)
        self.assertEqual(guard["task_sha256"], metadata["sha256"])
        self.assertEqual(
            guard["metadata_sha256"],
            hashlib.sha256((ROOT / "data/evaluation/metadata.json").read_bytes()).hexdigest(),
        )

    def test_every_final_task_has_a_reachable_and_scored_gold_purchase(self):
        with gzip.open(
            ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
            "rt",
            encoding="utf-8",
        ) as stream:
            products = json.load(stream)
        task_ids = [
            json.loads(line)["task_id"]
            for line in (ROOT / "data/evaluation/tasks.jsonl").read_text().splitlines()
        ]
        for task_id in task_ids:
            target = products[task_id]
            instruction = target["instructions"][0]
            product = deepcopy(target)
            product.update(
                {
                    "Title": target["title"],
                    "Description": target.get("full_description", ""),
                    "BulletPoints": [],
                    "Attributes": target.get("attribute", []),
                    "pricing": target.get("pricing", []),
                }
            )
            goal = {
                "asin": target["asin"],
                "category": target["category"],
                "price_upper": explicit_budget_from_instruction(
                    instruction["instruction"]
                ),
            }
            goal.update(compile_reward_features(instruction, target))
            selected, _ = candidate_options_for_evaluation(
                product, goal["required_options_by_key"]
            )
            price = resolve_variant_price(product, selected)
            result = evaluate_purchase(
                product, goal, selected_options=selected, price_resolution=price
            )
            self.assertFalse(goal["unresolved_option_requirements"], task_id)
            self.assertEqual(price["status"], "pass", task_id)
            if PRICE_HINT.search(instruction["instruction"]):
                self.assertIsNotNone(goal["price_upper"], task_id)
            self.assertEqual(result.reward_type, "gold_purchase", task_id)


if __name__ == "__main__":
    unittest.main()
