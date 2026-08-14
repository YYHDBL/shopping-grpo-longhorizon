import json
import unittest

from scripts.prepare_sft_curriculum import build_manifest


def _call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _row(task_id):
    asin = f"asin-{task_id}"
    names = ("search_products", "open_product", "select_option", "buy_now")
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": f"buy task {task_id}"},
        _call("search_products", {"query": "item"}, f"{task_id}-1"),
        {"role": "tool", "tool_call_id": f"{task_id}-1", "content": asin},
        _call("open_product", {"asin": asin}, f"{task_id}-2"),
        {
            "role": "tool",
            "tool_call_id": f"{task_id}-2",
            "content": f"asin: {asin}\nprice: 10",
        },
        _call("select_option", {"value": "red"}, f"{task_id}-3"),
        {"role": "tool", "tool_call_id": f"{task_id}-3", "content": "selected"},
        _call("buy_now", {}, f"{task_id}-4"),
        {"role": "tool", "tool_call_id": f"{task_id}-4", "content": "购买已完成。"},
    ]
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": name}} for name in names],
    }


def _label(task_id, difficulty, *, rewrite=False, compare=False):
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "trajectory_complexity": difficulty,
        "needs_query_rewrite": rewrite,
        "needs_candidate_comparison": compare,
    }


class PrepareSftCurriculumTests(unittest.TestCase):
    def test_builds_stable_atomic_buckets_and_cumulative_stages(self):
        rows = [_row(task_id) for task_id in range(1, 31)]
        labels = []
        for task_id in range(1, 11):
            labels.append(_label(task_id, "simple"))
        for task_id in range(11, 21):
            labels.append(_label(task_id, "medium"))
        for task_id in range(21, 26):
            labels.append(_label(task_id, "medium", compare=True))
        for task_id in range(26, 31):
            labels.append(_label(task_id, "hard", rewrite=True, compare=True))

        manifest = build_manifest(
            rows,
            labels,
            evaluation_ids=set(),
            seed=7,
            validation_ratio=0.2,
        )
        again = build_manifest(
            rows,
            labels,
            evaluation_ids=set(),
            seed=7,
            validation_ratio=0.2,
        )

        self.assertEqual(manifest, again)
        self.assertEqual(manifest["counts"]["rows"], 30)
        self.assertEqual(manifest["counts"]["train"], 24)
        self.assertEqual(manifest["counts"]["validation"], 6)
        self.assertEqual(manifest["stages"]["a"]["train_rows"], 8)
        self.assertEqual(manifest["stages"]["b"]["train_rows"], 16)
        self.assertEqual(manifest["stages"]["c"]["train_rows"], 24)
        train_ids = {
            task_id
            for bucket in manifest["buckets"].values()
            for task_id in bucket["train_task_ids"]
        }
        validation_ids = {
            task_id
            for bucket in manifest["buckets"].values()
            for task_id in bucket["validation_task_ids"]
        }
        self.assertFalse(train_ids & validation_ids)

    def test_rejects_evaluation_overlap_and_invalid_tool_arguments(self):
        row = _row(1)
        label = _label(1, "simple")
        with self.assertRaisesRegex(ValueError, "evaluation overlap"):
            build_manifest([row], [label], evaluation_ids={1})

        row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "not-json"
        with self.assertRaisesRegex(ValueError, "invalid tool arguments"):
            build_manifest([row], [label], evaluation_ids=set())

    def test_records_review_flags_without_dropping_valid_hard_cases(self):
        row = _row(1)
        label = _label(1, "hard", compare=True)

        manifest = build_manifest([row], [label], evaluation_ids=set())

        self.assertEqual(manifest["counts"]["rows"], 1)
        self.assertEqual(
            manifest["review_flags"]["candidate_comparison_under_evidenced"],
            [1],
        )

    def test_accepts_sanitized_terminal_observation_after_purchase(self):
        row = _row(1)
        row["messages"][-1]["content"] = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: terminal\n\n"
            "搜索功能是否可用: False\n可点击的按钮: []"
        )

        manifest = build_manifest(
            [row],
            [_label(1, "simple")],
            evaluation_ids=set(),
        )

        self.assertEqual(manifest["counts"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
