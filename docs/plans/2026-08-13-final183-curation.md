# Final-183 Curation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the audited Final-200 benchmark with one fair, leak-protected Final-183 task list and publish its evidence.

**Architecture:** Keep the existing `data/evaluation/tasks.jsonl` path so evaluation callers do not need compatibility flags. Update its metadata and packaged blind-ID guard atomically, then document the exclusion policy and archive-level model analysis in Markdown.

**Tech Stack:** JSONL, JSON metadata, Python `unittest`, Markdown, Git.

---

### Task 1: Lock the curated task contract

**Files:**
- Modify: `data/evaluation/tasks.jsonl`
- Modify: `data/evaluation/metadata.json`
- Modify: `src/shopping_grpo/resources/blind_final_task_ids.json`
- Modify: `src/shopping_grpo/resources/blind_guard.json`
- Modify: `src/shopping_grpo/evaluation/blind_guard.py`
- Test: `tests/test_evaluation_dataset.py`

1. Add a regression test for 183 unique IDs, the 17 exclusions, matching SHA-256 metadata, and matching packaged blind IDs.
2. Run the test and confirm it fails against Final-200.
3. Remove only the audited IDs from the canonical list and update its hash, metadata, and guard resources together.
4. Re-run the regression test and the existing blind-guard tests.

### Task 2: Publish the curation and archive analysis

**Files:**
- Create: `docs/evaluation-dataset.md`
- Create: `docs/evaluation-updates.md`
- Modify: `docs/README.md`
- Modify: `README.md`
- Modify: `data/README.md`

1. Document Final-183 as the only current benchmark, including every exclusion and its reason.
2. Add an append-only update format and record the Final-200 trajectory audit, per-model results, shared bad cases, and the Final-183 re-score.
3. Mark Final-200 reports as historical so old and new denominators are not mixed.

### Task 3: Verify and publish

**Files:**
- Verify all changed files

1. Run focused tests and hash/overlap checks.
2. Review the staged diff for unintended user changes.
3. Commit only the curation, docs, and already-requested report automation changes; push `main` to `origin`.
