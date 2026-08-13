# Evaluation Report Automation Implementation Plan

**Goal:** Make every Shopping GRPO evaluation directory produce a reusable, model-agnostic HTML report from its `summary.json` and `trajectories.jsonl`.

**Architecture:** Parameterize the existing self-contained report builder with `--run-dir` and derive the model/protocol labels from `summary.json`. Keep task-level aggregation in Python, embed only compact data in the HTML, and invoke the builder after the normal evaluation command completes. An explicit report command remains available for historical runs.

**Tech Stack:** Python standard library, existing HTML/CSS/JavaScript template, Bash wrapper, unittest.

---

### Tasks

1. Add a failing test proving a temporary evaluation directory can be parsed and its model name appears in generated report data.
2. Parameterize `scripts/build_glm_report.py` with `--run-dir` and `--output`, replacing hard-coded GLM-5.2 paths and labels.
3. Add `scripts/report.sh NAME` as the short one-command entry point for existing output directories.
4. Update `scripts/evaluate.sh` to generate `report.html` after evaluation completes.
5. Run the focused test, existing benchmark CLI test, Python compilation, and regenerate the existing GLM-5.2 report as a compatibility check.
