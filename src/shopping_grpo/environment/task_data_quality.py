"""Offline task-data checks. Findings never silently rewrite actor inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy

AUDIT_VERSION = "task-data-quality-v1"
REQUEST = re.compile(
    r"我想|我需要|帮我|需要(?:找)?一款|找一款|推荐一款|预算|价格(?:控制|不超过|在)"
)


def goal_index(products):
    """Mirror the unfiltered, unshuffled loader's stable task ordering."""
    rows, seen = [], set()
    for pi, product in enumerate(products):
        asin = product["asin"]
        if asin == "nan" or len(asin) > 20 or asin in seen:
            continue
        seen.add(asin)
        for ii, instruction in enumerate(product.get("instructions", [])):
            if not instruction.get("attributes", []):
                continue
            rows.append((pi, ii))
    return rows


def audit_task(product, instruction):
    findings = []
    text = instruction.get("instruction")
    attrs = product.get("attribute")
    goal_attrs = instruction.get("attributes")

    def add(code, severity, field, evidence):
        findings.append({"code": code, "severity": severity, "field": field, "evidence": evidence})

    if not isinstance(text, str) or not text.strip():
        add("missing_instruction", "error", "instruction", text)
        text = ""
    for name, values in [("attribute", attrs), ("attributes", goal_attrs)]:
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            add("invalid_attribute_schema", "error", name, values)
    attrs = attrs if isinstance(attrs, list) else []
    request_attrs = [a for a in attrs if isinstance(a, str) and REQUEST.search(a)]
    if request_attrs:
        add("request_in_product_attributes", "review", "attribute", request_attrs)
    # Shortness alone is a weak signal. A swap requires request-like attributes,
    # a terse keyword list, and identical product/goal attribute fields.
    if text and len(text.strip()) < 12:
        add("short_instruction", "review", "instruction", text)
    if (
        request_attrs
        and attrs == goal_attrs
        and 0 < len(text.strip()) <= 32
        and not REQUEST.search(text)
        and len("".join(attrs)) > 2 * len(text)
    ):
        add(
            "suspected_instruction_attribute_swap",
            "error",
            "instruction|attribute|attributes",
            {"instruction": text, "attributes": attrs},
        )
    if "\ufffd" in text or "??" in text:
        add("damaged_text_marker", "review", "instruction", text)
    return findings


def audit_scope(products, task_ids, expected_instructions=None):
    ids = list(task_ids)
    if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("scope must contain unique nonnegative integer task IDs")
    index = goal_index(products)
    records = []
    for tid in sorted(ids):
        if tid >= len(index):
            raise ValueError(f"task ID outside source index: {tid}")
        pi, ii = index[tid]
        product = products[pi]
        instruction = product["instructions"][ii]
        if expected_instructions is not None and (
            tid not in expected_instructions
            or instruction["instruction"] != expected_instructions[tid]
        ):
            raise ValueError(f"frozen instruction/index mismatch: {tid}")
        records.append(
            {
                "task_id": tid,
                "product_index": pi,
                "instruction_index": ii,
                "asin": product["asin"],
                "instruction": instruction["instruction"],
                "findings": audit_task(product, instruction),
            }
        )
    counts = Counter(f["code"] for r in records for f in r["findings"])
    return {
        "version": AUDIT_VERSION,
        "scope_count": len(ids),
        "goal_count": len(index),
        "flagged_tasks": sum(bool(r["findings"]) for r in records),
        "blocking_tasks": [
            r["task_id"] for r in records if any(f["severity"] == "error" for f in r["findings"])
        ],
        "finding_counts": dict(counts),
        "records": records,
        "limitations": "Pattern-based screening only; a clean result is not proof of semantic correctness. No automatic repair.",
    }


def source_digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_reviewed_repairs(products, source_sha256, repair_manifest, scope):
    """Apply explicit field replacements only to a copy, with stale-data guards."""
    if repair_manifest.get("source_sha256") != source_sha256:
        raise ValueError("repair source hash mismatch")
    repaired = deepcopy(products)
    index = goal_index(products)
    changed = set()
    for patch in repair_manifest["repairs"]:
        tid = patch["task_id"]
        if tid not in scope or tid in changed:
            raise ValueError("repair outside audit scope or duplicate task")
        if not patch.get("provenance"):
            raise ValueError("repair provenance required")
        pi, ii = index[tid]
        product = repaired[pi]
        # Shared product attributes affect all instructions; require an explicit
        # dedicated migration instead of silently changing unreviewed tasks.
        if len(product["instructions"]) != 1:
            raise ValueError("shared product attributes require separate review")
        instruction = product["instructions"][ii]
        if product["asin"] != patch["asin"]:
            raise ValueError("repair ASIN mismatch")
        before = {
            "instruction": instruction["instruction"],
            "attributes": instruction["attributes"],
            "attribute": product["attribute"],
        }
        if before != patch["before"]:
            raise ValueError("repair field precondition mismatch")
        after = patch["after"]
        if (
            set(after) != set(before)
            or not isinstance(after["instruction"], str)
            or not after["instruction"].strip()
        ):
            raise ValueError("repair must provide exactly three valid replacement fields")
        for name in ("attribute", "attributes"):
            if (
                not isinstance(after[name], list)
                or not after[name]
                or any(not isinstance(v, str) or not v for v in after[name])
            ):
                raise ValueError("repair attributes must be nonempty string lists")
        instruction["instruction"] = after["instruction"]
        instruction["attributes"] = list(after["attributes"])
        product["attribute"] = list(after["attribute"])
        changed.add(tid)
    if goal_index(repaired) != index:
        raise ValueError("repair changed task ID ordering")
    return repaired


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="Explicit JSONL scope: task_id and optional instruction (without prefix)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repairs", type=Path)
    parser.add_argument("--repaired-products", type=Path)
    parser.add_argument("--fail-on", choices=["error", "any", "none"], default="error")
    args = parser.parse_args(argv)
    if bool(args.repairs) != bool(args.repaired_products):
        parser.error("--repairs and --repaired-products must be supplied together")
    # Refuse every input/output alias before writing anything.
    inputs = [args.products, args.tasks] + ([args.repairs] if args.repairs else [])
    outputs = [args.output] + ([args.repaired_products] if args.repaired_products else [])
    paths = [p.resolve() for p in inputs + outputs]
    if len(set(paths)) != len(paths):
        parser.error("input/output paths must be distinct")
    if any(p.exists() for p in outputs):
        parser.error("output exists; choose a fresh versioned destination")
    data = args.products.read_bytes()
    products = json.loads(data)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    ids = [r["task_id"] for r in tasks]
    expected = None
    if any("instruction" in r for r in tasks):
        if not all("instruction" in r for r in tasks):
            parser.error("instruction snapshots must be present for every scoped task or none")
        expected = {r["task_id"]: r["instruction"] for r in tasks}
    before = audit_scope(products, ids, expected)
    report = {
        "source_sha256": source_digest(data),
        "scope_sha256": source_digest(args.tasks.read_bytes()),
        "before": before,
    }
    final = before
    if args.repairs:
        manifest = json.loads(args.repairs.read_text())
        repaired = apply_reviewed_repairs(products, report["source_sha256"], manifest, set(ids))
        final = audit_scope(repaired, ids)
        payload = (json.dumps(repaired, ensure_ascii=False, indent=2) + "\n").encode()
        report.update(
            {
                "repair_manifest_sha256": source_digest(args.repairs.read_bytes()),
                "after": final,
                "repaired_products_sha256": source_digest(payload),
                "activation": "candidate_only; original dataset and running environment unchanged",
            }
        )
        args.repaired_products.parent.mkdir(parents=True, exist_ok=True)
        with args.repaired_products.open("xb") as handle:
            handle.write(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "scope_count": final["scope_count"],
                "blocking_tasks": final["blocking_tasks"],
                "flagged_tasks": final["flagged_tasks"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return (
        int(
            (args.fail_on == "error" and bool(final["blocking_tasks"]))
            or (args.fail_on == "any" and bool(final["flagged_tasks"]))
        )
        * 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
