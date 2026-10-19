import copy
import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.environment.task_data_quality import (
    apply_reviewed_repairs,
    audit_scope,
    audit_task,
    goal_index,
    main,
)


def product(asin="a"):
    return {
        "asin": asin,
        "attribute": ["需要一款红色杯子", "预算30元。"],
        "instructions": [
            {"instruction": "红色，陶瓷", "attributes": ["需要一款红色杯子", "预算30元。"]}
        ],
        "customization_options": {"颜色": [{"value": "红色", "price": 30}]},
    }


def patch(p):
    i = p["instructions"][0]
    return {
        "source_sha256": "hash",
        "repairs": [
            {
                "task_id": 0,
                "asin": p["asin"],
                "before": {
                    "instruction": i["instruction"],
                    "attributes": i["attributes"],
                    "attribute": p["attribute"],
                },
                "after": {
                    "instruction": "需要一款红色杯子，预算30元。",
                    "attributes": ["红色", "陶瓷"],
                    "attribute": ["红色", "陶瓷"],
                },
                "provenance": "Manually reviewed source field transposition; not generated from hidden target options.",
            }
        ],
    }


class TaskDataQualityTest(unittest.TestCase):
    def test_short_valid_request_is_not_a_confirmed_swap(self):
        p = {"attribute": ["红色"], "instructions": []}
        codes = {
            f["code"]
            for f in audit_task(p, {"instruction": "买红色杯子", "attributes": ["红色"]})
            if f["severity"] == "error"
        }
        self.assertEqual(codes, set())

    def test_transposition_is_blocked_but_missing_chars_not_reconstructed(self):
        p = product()
        self.assertIn(
            "suspected_instruction_attribute_swap",
            [f["code"] for f in audit_task(p, p["instructions"][0])],
        )
        i = {"instruction": "杯子??红色", "attributes": ["红色"]}
        self.assertIn(
            "damaged_text_marker", [f["code"] for f in audit_task({"attribute": ["红色"]}, i)]
        )
        self.assertEqual(i["instruction"], "杯子??红色")

    def test_index_respects_loader_skips(self):
        p = product()
        duplicate = product()
        invalid = product("nan")
        empty = product("b")
        empty["instructions"][0]["attributes"] = []
        last = product("c")
        self.assertEqual(goal_index([invalid, p, duplicate, empty, last]), [(1, 0), (4, 0)])
        self.assertEqual(
            audit_scope([invalid, p, duplicate, empty, last], [1])["records"][0]["asin"], "c"
        )

    def test_scope_and_instruction_snapshot_fail_closed(self):
        for ids in ([0, 0], [-1], [1], [True]):
            with self.assertRaises(ValueError):
                audit_scope([product()], ids)
        with self.assertRaisesRegex(ValueError, "snapshot|mismatch"):
            audit_scope([product()], [0], {0: "different"})

    def test_repair_leaves_original_and_unscoped_products_unchanged(self):
        p = product()
        other = product("b")
        original = copy.deepcopy([p, other])
        changed = apply_reviewed_repairs([p, other], "hash", patch(p), {0})
        self.assertEqual([p, other], original)
        self.assertEqual(changed[1], other)
        self.assertEqual(changed[0]["customization_options"], p["customization_options"])
        self.assertEqual(audit_scope(changed, [0])["blocking_tasks"], [])

    def test_stale_hash_and_field_mismatch_and_out_of_scope_rejected(self):
        p = product()
        with self.assertRaisesRegex(ValueError, "hash"):
            apply_reviewed_repairs([p], "wrong", patch(p), {0})
        with self.assertRaisesRegex(ValueError, "scope"):
            apply_reviewed_repairs([p], "hash", patch(p), set())
        m = patch(p)
        m["repairs"][0]["before"]["instruction"] = "stale"
        with self.assertRaisesRegex(ValueError, "precondition"):
            apply_reviewed_repairs([p], "hash", m, {0})

    def test_shared_product_fields_cannot_change_unreviewed_instruction(self):
        p = product()
        p["instructions"].append(copy.deepcopy(p["instructions"][0]))
        with self.assertRaisesRegex(ValueError, "shared"):
            apply_reviewed_repairs([p], "hash", patch(p), {0})

    def test_cli_reports_blockers_without_overwriting_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            source = d / "products.json"
            scope = d / "tasks.jsonl"
            out = d / "audit.json"
            source.write_text(json.dumps([product()]))
            scope.write_text('{"task_id":0}\n')
            before = source.read_bytes()
            argv = ["--products", str(source), "--tasks", str(scope), "--output", str(out)]
            self.assertEqual(main(argv), 2)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(SystemExit):
                main(argv)
            with self.assertRaises(SystemExit):
                main(["--products", str(source), "--tasks", str(scope), "--output", str(source)])
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
