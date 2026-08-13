# Final-200 Replenishment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a 200-task audited, train-isolated benchmark without reintroducing any rejected source task.

**Architecture:** Keep the existing evaluation path and packaged blind-ID guard. Remove every source task with an unreachable target, contradictory gold, or an unscored price requirement; select constraint-stratified replacements from the frozen ShopSimulator pool and validate them before atomically updating the manifest, guard and documentation.

**Tech Stack:** Python standard library, ShopSimulator goal data, JSONL/JSON, `unittest`, Markdown, Git.

---

### Task 1: Define and test the replacement contract

**Files:**
- Modify: `tests/test_evaluation_dataset.py`
- Modify: `data/evaluation/tasks.jsonl`
- Modify: `data/evaluation/metadata.json`
- Modify: `src/shopping_grpo/resources/blind_final_task_ids.json`
- Modify: `src/shopping_grpo/resources/blind_guard.json`

1. Add a regression assertion for 200 unique task IDs, the permanent exclusion of the 17 rejected IDs, matching manifest hash and matching blind guard.
2. Select audited candidate IDs outside all active training pools, preserving the removed constraint-count distribution.
3. Validate each candidate’s goal options against its target product, the explicit budget against an available target variant price, and instruction/annotation consistency.
4. Update the canonical list, metadata and packaged guard together.

### Task 2: Publish the selection evidence

**Files:**
- Modify: `docs/evaluation-dataset.md`
- Modify: `docs/evaluation-updates.md`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `data/README.md`
- Modify: `docs/README.md`
- Modify: `docs/evaluation.md`

1. Document all 17 replacements with their task IDs, plain-language query summaries and the checks they passed.
2. State the Final-200 quality contract and its remaining limits: manual audit catches label/parser contradictions, while it does not prove every valid alternative is rewarded equally.
3. Record the denominator change as a benchmark version change; do not relabel old Final-183 recomputations as new rollouts.

### Task 3: Verify and publish

**Files:**
- Verify all changed files

1. Run focused dataset, guard and benchmark tests.
2. Run a standalone replacement audit and all training/evaluation overlap checks.
3. Review the staged diff, commit only related files and push `main` to `origin`.
