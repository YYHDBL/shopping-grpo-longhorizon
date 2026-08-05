"""End-to-end CPU tests for reproducible Feed dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.datasets import (
    ARTIFACT_PATHS,
    CURRICULUM_STAGES,
    MAX_FEED_LENGTH,
    POLICY_NAMES,
    SFT_MESSAGE_CHAR_BUDGET,
    SFT_SHORT_WINDOW_MAX_STEPS,
    _sft_rows,
    generate_episode_splits,
    generate_feed_artifacts,
)
from shopping_grpo.feed.manifest import (
    audit_split_isolation,
    canonical_json,
    verify_manifest,
)
from shopping_grpo.feed.schema import Product, iter_jsonl
from shopping_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example
from scripts.prepare_feed_dpo_data import prepare_directory as prepare_dpo_directory


class _CharacterTokenizer:
    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>" + (message.get("content") or "")
            for call in message.get("tool_calls") or []:
                function = call["function"]
                text += f"[tool={function['name']} args={function['arguments']}]"
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _catalog() -> ProductCatalog:
    products = []
    for category_index in range(4):
        complement_category = (category_index + 1) % 4
        for item_index in range(4):
            products.append(
                Product(
                    product_id=f"p{category_index}{item_index}",
                    title=f"category {category_index} item {item_index}",
                    category=f"category-{category_index}",
                    price=float(10 + 10 * item_index + category_index),
                    attributes=(f"category-attribute-{category_index}", f"style-{item_index}"),
                    rating=3.0 + item_index / 3.0,
                    popularity=float(item_index),
                    review_summary="public review",
                    complement_product_ids=(f"p{complement_category}0",),
                )
            )
    return ProductCatalog(products)


def _assert_tool_pairs(test: unittest.TestCase, messages) -> None:
    pending = set()
    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call["id"]
                test.assertNotIn(call_id, pending)
                pending.add(call_id)
                test.assertIsInstance(
                    json.loads(call["function"]["arguments"]), dict
                )
        elif message["role"] == "tool":
            test.assertIn(message["tool_call_id"], pending)
            pending.remove(message["tool_call_id"])
    test.assertFalse(pending)


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_nested_keys(nested) for nested in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_nested_keys(nested) for nested in value))
    return set()


def _assert_delta_tool_replies(test: unittest.TestCase, messages) -> None:
    for message in messages:
        if message["role"] != "tool":
            continue
        payload = json.loads(message["content"])
        test.assertEqual(payload["payload_version"], "feed-tool-delta-v1")
        test.assertTrue(payload["ok"])
        test.assertEqual(payload["tool"], message["name"])
        test.assertIn("result", payload)
        test.assertIn("state_delta", payload)
        test.assertNotIn("observation", payload)
        test.assertNotIn("persona", payload["state_delta"])
        test.assertNotIn("recent_events", payload["state_delta"])
        if message["name"] == "commit_recommendation":
            test.assertNotIn("observation", payload["result"])
            test.assertIn("current_video", payload["state_delta"])
            test.assertTrue(
                {"observation", "persona", "recent_events"}.isdisjoint(
                    _nested_keys(payload)
                )
            )


class FeedDatasetTest(unittest.TestCase):
    def test_feed_length_is_capped_at_long_horizon_contract(self):
        with self.assertRaisesRegex(ValueError, "24 to 48"):
            generate_episode_splits(
                _catalog(),
                episodes=3,
                feed_length=MAX_FEED_LENGTH + 1,
                seed=17,
            )

    def test_sft_long_horizon_row_is_bounded_atomic_and_complete(self):
        seed_record = generate_episode_splits(
            _catalog(),
            episodes=3,
            feed_length=24,
            seed=19,
        )["train"][0]
        chunks = []
        for step in range(24):
            call_id = f"synthetic-commit-{step:03d}"
            chunks.append(
                [
                    {
                        "role": "user",
                        "content": f"step-{step:03d}|完整公开状态检查点",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "commit_recommendation",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "commit_recommendation",
                        "content": canonical_json(
                            {"done": False, "public_delta": "证据" * 3_000}
                        ),
                    },
                ]
            )

        rows = _sft_rows(seed_record, chunks)
        for row in rows:
            _assert_tool_pairs(self, row["messages"])
            self.assertEqual(
                sum(message["role"] == "user" for message in row["messages"]),
                1,
            )
            serialized_chars = len(canonical_json(row["messages"]))
            self.assertLessEqual(serialized_chars, SFT_MESSAGE_CHAR_BUDGET)
            self.assertEqual(row["serialized_message_chars"], serialized_chars)
            self.assertEqual(row["message_char_budget"], SFT_MESSAGE_CHAR_BUDGET)

        short = next(
            row for row in rows if row["curriculum_stage"] == "B_short_window"
        )
        self.assertEqual(short["window_start_step"], 0)
        self.assertLessEqual(short["window_steps"], SFT_SHORT_WINDOW_MAX_STEPS)

        long_rows = [
            row for row in rows if row["curriculum_stage"] == "C_long_horizon"
        ]
        self.assertEqual(len(long_rows), 1)
        self.assertEqual(long_rows[0]["window_index"], 0)
        self.assertEqual(long_rows[0]["window_count"], 1)
        self.assertEqual(long_rows[0]["window_start_step"], 0)
        self.assertEqual(long_rows[0]["window_end_step_exclusive"], 24)
        self.assertEqual(long_rows[0]["window_steps"], 24)

    def test_seed_splits_are_isolated_and_have_all_candidate_roles(self):
        catalog = _catalog()
        splits = generate_episode_splits(
            catalog,
            episodes=3,
            feed_length=24,
            seed=17,
        )

        summary = audit_split_isolation(splits)
        self.assertEqual(
            {split: data["rows"] for split, data in summary.items()},
            {"test": 1, "train": 1, "validation": 1},
        )
        episode_ids = set()
        persona_ids = set()
        for rows in splits.values():
            for record in rows:
                self.assertNotIn(record.episode_id, episode_ids)
                self.assertNotIn(record.persona.persona_id, persona_ids)
                episode_ids.add(record.episode_id)
                persona_ids.add(record.persona.persona_id)
                self.assertEqual(len(record.videos), 24)
                self.assertGreaterEqual(len(record.metadata["categories"]), 3)
                self.assertLessEqual(len(record.metadata["categories"]), 5)
                for video in record.videos:
                    roles = record.metadata["candidate_roles_by_video"][video.video_id]
                    self.assertEqual(
                        set(roles),
                        {
                            "strong_relevant",
                            "hard_negative",
                            "cheaper_alternative",
                            "complement",
                            "unrelated",
                        },
                    )
                    role_ids = [identifier for ids in roles.values() for identifier in ids]
                    self.assertEqual(len(role_ids), len(set(role_ids)))
                    self.assertTrue(set(role_ids).issubset(record.product_ids))
                    strong = catalog.require(roles["strong_relevant"][0])
                    cheaper = catalog.require(roles["cheaper_alternative"][0])
                    hard = catalog.require(roles["hard_negative"][0])
                    unrelated = catalog.require(roles["unrelated"][0])
                    self.assertLess(cheaper.price, strong.price)
                    self.assertEqual(hard.category, strong.category)
                    self.assertNotEqual(unrelated.category, strong.category)

    def test_five_artifacts_manifest_curriculum_and_sft_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifacts"
            manifest = generate_feed_artifacts(
                _catalog(),
                output,
                episodes=3,
                feed_length=24,
                seed=23,
            )

            verified = verify_manifest(output)
            self.assertEqual(
                manifest["manifest_content_sha256"],
                verified["manifest_content_sha256"],
            )
            for directory in ARTIFACT_PATHS.values():
                for split in ("train", "validation", "test"):
                    self.assertTrue((output / directory / f"{split}.jsonl").is_file())

            logs = list(iter_jsonl(output / "mixed_policy_logs" / "train.jsonl"))
            self.assertEqual({row["behavior_policy"] for row in logs}, set(POLICY_NAMES))
            self.assertTrue(
                all("shopping_grpo.feed.policies" in row["policy_source"] for row in logs)
            )
            teacher = next(row for row in logs if row["behavior_policy"] == "Teacher")
            teacher_strategies = {
                transition["action"]["strategy"]
                for transition in teacher["transitions"]
            }
            self.assertTrue(
                {"bundle", "cheaper_alternative", "review_summary"}.issubset(
                    teacher_strategies
                )
            )
            for row in logs:
                public_summary = json.dumps(row["summary"], ensure_ascii=False)
                self.assertNotIn("terminal_fatigue", public_summary)
                self.assertNotIn("terminal_satisfaction", public_summary)
                self.assertTrue(row["evaluator_summary"]["evaluator_only"])
                self.assertIn("terminal_fatigue", row["evaluator_summary"])
                self.assertIn("terminal_satisfaction", row["evaluator_summary"])
                self.assertNotIn(
                    "candidate_roles_by_video",
                    json.dumps(row, ensure_ascii=False),
                )

            sft_rows = list(iter_jsonl(output / "sft_trajectories" / "train.jsonl"))
            self.assertEqual(
                {row["curriculum_stage"] for row in sft_rows},
                set(CURRICULUM_STAGES),
            )
            for stage in CURRICULUM_STAGES:
                staged = list(
                    iter_jsonl(
                        output / "sft_trajectories" / f"train.{stage}.jsonl"
                    )
                )
                self.assertTrue(staged)
                self.assertEqual({row["curriculum_stage"] for row in staged}, {stage})
            for row in sft_rows:
                _assert_tool_pairs(self, row["messages"])
                _assert_delta_tool_replies(self, row["messages"])
                self.assertEqual(
                    sum(message["role"] == "user" for message in row["messages"]),
                    1,
                )
                serialized_chars = len(canonical_json(row["messages"]))
                self.assertLessEqual(serialized_chars, SFT_MESSAGE_CHAR_BUDGET)
                self.assertEqual(row["serialized_message_chars"], serialized_chars)
                serialized = json.dumps(row["messages"], ensure_ascii=False)
                self.assertNotIn("related_product_ids", serialized)
                self.assertNotIn("candidate_roles", serialized)

            short_rows = [
                row
                for row in sft_rows
                if row["curriculum_stage"] == "B_short_window"
            ]
            self.assertEqual(len(short_rows), 1)
            self.assertEqual(short_rows[0]["window_start_step"], 0)
            self.assertLessEqual(
                short_rows[0]["window_steps"], SFT_SHORT_WINDOW_MAX_STEPS
            )

            long_rows = [
                row
                for row in sft_rows
                if row["curriculum_stage"] == "C_long_horizon"
            ]
            cursor = 0
            for row in long_rows:
                self.assertEqual(row["window_start_step"], cursor)
                cursor = row["window_end_step_exclusive"]
            self.assertEqual(cursor, 24)

            action_contract = next(
                row
                for row in sft_rows
                if row["curriculum_stage"] == "A_action_contract"
            )
            supervised = build_supervised_example(
                action_contract["messages"],
                action_contract["tools"],
                _CharacterTokenizer(),
                max_length=1_000_000,
            )
            self.assertIsNotNone(supervised)
            self.assertTrue(
                any(label != IGNORE_INDEX for label in supervised["labels"])
            )

            preferences = list(
                iter_jsonl(output / "preference_pairs" / "train.jsonl")
            )
            self.assertEqual(len(preferences), 1)
            self.assertTrue(preferences[0]["common_random_numbers"])
            self.assertEqual(preferences[0]["counterfactual_method"], "common_random_numbers")
            self.assertIn(
                "recorded_action_replay",
                preferences[0]["counterfactual_policy_source"],
            )
            _assert_tool_pairs(self, preferences[0]["prompt"])
            _assert_delta_tool_replies(self, preferences[0]["prompt"])
            self.assertEqual(
                sum(
                    message["role"] == "user"
                    for message in preferences[0]["prompt"]
                ),
                1,
            )
            self.assertNotEqual(
                preferences[0]["chosen_action"],
                preferences[0]["rejected_action"],
            )
            self.assertGreaterEqual(preferences[0]["return_margin"], 0.0)
            self.assertIn("tools", preferences[0])
            chosen_call = preferences[0]["chosen"]["tool_calls"][0]["id"]
            rejected_call = preferences[0]["rejected"]["tool_calls"][0]["id"]
            self.assertEqual(chosen_call, rejected_call)
            self.assertNotIn("chosen", chosen_call.casefold())
            self.assertNotIn("rejected", chosen_call.casefold())

            tasks = list(iter_jsonl(output / "online_rl_tasks" / "train.jsonl"))
            self.assertEqual(len(tasks), 1)
            self.assertIn("episode_seed", tasks[0])
            self.assertIn("catalog_products", tasks[0])
            self.assertNotIn("catalog_path", tasks[0])
            self.assertEqual(tasks[0]["extra_info"]["split"], "train")
            self.assertEqual(
                set(manifest["curriculum_stage_counts"]["train"]),
                set(CURRICULUM_STAGES),
            )
            self.assertFalse(str(manifest["source_catalog"]["path"] or "").startswith("/"))
            dpo = prepare_dpo_directory(
                output / "preference_pairs",
                output / "dpo",
                inspect_only=True,
            )
            self.assertTrue(dpo["inspect_only"])
            self.assertEqual(dpo["splits"]["train"]["rows"], 1)

    def test_same_inputs_produce_identical_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            kwargs = {
                "episodes": 3,
                "feed_length": 24,
                "seed": 29,
                "prefer_external_policies": False,
            }
            generate_feed_artifacts(_catalog(), first, **kwargs)
            generate_feed_artifacts(_catalog(), second, **kwargs)

            for directory in ARTIFACT_PATHS.values():
                for split in ("train", "validation", "test"):
                    relative = Path(directory) / f"{split}.jsonl"
                    self.assertEqual(
                        (first / relative).read_bytes(),
                        (second / relative).read_bytes(),
                    )
            self.assertEqual(
                (first / "manifest.json").read_bytes(),
                (second / "manifest.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
