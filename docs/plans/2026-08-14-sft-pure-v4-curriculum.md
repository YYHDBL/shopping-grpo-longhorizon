# SFT Pure V4 Curriculum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Pure V4 the audited SFT source and provide a deterministic three-stage curriculum that runs end-to-end with one server command.

**Architecture:** A preparation script validates all source rows, creates a fixed skill-stratified train/development manifest, and records review-only process flags. The existing SFT trainer gains task-ID filtering, while a thin curriculum runner trains and merges cumulative stages A, B, and C without duplicating the 26 MB trajectory file.

**Tech Stack:** Python standard library, existing Transformers/PEFT launchers, JSON, Bash, `unittest`.

---

### Task 1: Build the deterministic curriculum manifest

**Files:**
- Create: `scripts/prepare_sft_curriculum.py`
- Create: `tests/test_prepare_sft_curriculum.py`
- Create: `data/sft_curriculum/manifest.json`

**Step 1:** Write failing tests for hard validation, stable bucket splitting, zero evaluation overlap, cumulative stage counts, and review flags.

**Step 2:** Run `python -m unittest tests.test_prepare_sft_curriculum -v` and confirm the imports or assertions fail because the builder does not exist.

**Step 3:** Implement a standard-library builder that reads Pure V4 rows and labels, validates the quality contract, splits each atomic bucket by seeded task hash, and writes one manifest containing source hashes and task IDs.

**Step 4:** Run the focused test and generate the checked-in manifest from repository data. Verify totals `1192 = 1073 train + 119 development` and bucket counts `256/28`, `543/60`, `274/31`.

### Task 2: Filter the source dataset by curriculum IDs

**Files:**
- Modify: `src/shopping_grpo/training/sft/dataset.py`
- Modify: `scripts/train_lora_sft.py`
- Modify: `tests/test_sft_training.py`
- Modify: `tests/test_train_lora_sft_cli.py`

**Step 1:** Write failing tests showing that requested task IDs are filtered before tokenization and that the CLI accepts one manifest plus a stage selector.

**Step 2:** Run the focused tests and confirm failure for the missing API/arguments.

**Step 3:** Add optional task-ID filtering to the loader and manifest/stage CLI arguments. Preserve existing behavior when neither argument is supplied.

**Step 4:** Run all SFT dataset and CLI tests.

### Task 3: Run and merge three cumulative stages

**Files:**
- Create: `scripts/run_sft_curriculum.py`
- Create: `scripts/sft_curriculum.sh`
- Create: `tests/test_sft_curriculum.py`

**Step 1:** Write failing tests for manifest expansion, A/B/C command construction, previous-stage merged model handoff, dry-run behavior, and start/stop stage validation.

**Step 2:** Run the tests and confirm failure because the runner does not exist.

**Step 3:** Implement a thin subprocess orchestrator that calls the existing trainer and merger. Do not duplicate model-loading or training logic.

**Step 4:** Verify the dry-run prints all six commands and that the final merged path is `outputs/models/sft-curriculum/stage-c/merged`.

### Task 4: Document direct server use

**Files:**
- Modify: `docs/sft.md`
- Modify: `data/sft_pure_v4/README.md`
- Create: `data/sft_curriculum/README.md`

**Step 1:** Document the one-command launch, stage sizes, outputs, resume examples, monitoring flags, final GRPO checkpoint, and why Final-200 is not used for checkpoint selection.

**Step 2:** Check every command against `--help` or `--dry-run` output.

### Task 5: Verify the package

**Files:**
- Verify all files above.

**Step 1:** Regenerate the manifest and require a zero diff.

**Step 2:** Run the focused curriculum suite and the existing SFT/guard suites.

**Step 3:** Run Python compilation, shell syntax checks, `git diff --check`, and a server dry-run.

**Step 4:** Review the final diff and commit only curriculum-related files.
