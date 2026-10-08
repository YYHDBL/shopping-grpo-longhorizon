# Component evaluation evidence

The completed evaluation supports separating mechanisms, not claiming that every
component improves success. The explicit-tool reference S scored **130/200 (65%)**.
Removing B or C was associated with fewer successes; removing E or G was not.
These are single greedy runs with an infrastructure confound described below.
The refactored optional adapter in this PR has CPU validation, **not a new model
evaluation**. Historical results are evidence for its design, not a measurement of
this exact Git revision.

## Protocol and scoring

All completed conditions use the same fixed 200 task IDs, Stage-C model, unchanged
actor instructions, greedy decoding (temperature 0, top-p 1), and 35 executed-tool
steps. There is no training or model merge in this comparison. S and all seven
leave-one-component-out conditions disable automatic Description/Attributes/Reviews
observations. Public details must be acquired through explicit tools. A changes
the system prompt; F changes catalog/index snapshots, not the frozen actor requests.
The original-environment/prompt-only run is a separate historical control, not S−A.

Strict success requires a complete terminal result (`done`, `over`, purchase),
original Reward v3 `gold_purchase`, and `reward_valid=true`. Under reviewed scoring,
use the preserved full `legacy_result`, not the review-policy termination tag.
Infrastructure-invalid trajectories stay in the 200-task denominator and are
reported separately. The 802 new trajectories have unique expected IDs with no
missing/duplicate records; the runtime audit found zero LoRA delta and no optimizer
updates or saved checkpoints. Existing incomplete requests 14671 and 2717 stay in
the list; no requirements are reconstructed from hidden targets.

## Historical controls

| Condition | Strict successes | Rate | Infrastructure invalid | Interpretation |
|---|---:|---:|---:|---|
| Original environment and prompt | 118/200 | 59% | 0 | Original reference |
| Enhanced environment and prompt, automatic details on | 136/200 | 68% | 0 | Changes information acquisition; excluded from proposed default profile |
| Original environment + enhanced prompt only | 113/200 | 56.5% | 1 | Prompt alone did not reproduce the combined improvement |
| S: enhanced profile, automatic details off | 130/200 | 65% | 0 | Reference for component removals |

The prompt-only exception was task 10807 (`float(None)` during search). Excluding
that task from both paired controls gives 113/199 versus 117/199, not evidence for
a prompt-only improvement. The earlier enhanced-environment/original-prompt attempt
was stopped with zero completed trajectories; it contributes no score and is not S−A.
The 68% result must not be presented as an explicit-tool result.

## All component removals

“Gain/loss” counts task-level successes gained/lost relative to S. Percentage-point
change is the removal condition minus S; it is not an additive component effect.

| Condition | Strict successes | Rate | Change (pp) | Gain/loss | Invalid | Search implementation |
|---|---:|---:|---:|---:|---:|---|
| S | 130/200 | 65% | — | — | 0 | Before serialization |
| S−A: original prompt | 126/200 | 63% | −2.0 | 18/22 | 0 | Serialized |
| S−B: no extra purchase checks | 123/200 | 61.5% | −3.5 | 5/12 | 0 | Before serialization |
| S−C: original interface and feedback | 120/200 | 60% | −5.0 | 5/15 | 0 | Serialized |
| S−D: original observation budgets | 124/200 | 62% | −3.0 | 12/18 | 0 | Serialized |
| S−E: original context handling | 131/200 | 65.5% | +0.5 | 4/3 | 0 | Serialized |
| S−F: original catalog/index, initial run | 126/200 | 63% | −2.0 | 4/8 | 2 | Before serialization |
| S−F: two invalid tasks replaced | 128/200 | 64% | −1.0 | 4/6 | 0 | Mixed provenance: 198 old + 2 serialized |
| S−G: original purchase and eligibility policy | 131/200 | 65.5% | +0.5 | 5/4 | 0 | Before serialization |

## What each component does, and what the runs show

**A — verification prompt.** Structures search, evidence collection, option
selection and final checks. S−A loses 22 successes and gains 18, a small net loss
with substantial task turnover. Together with the prompt-only control, this supports
context-dependent prompt behavior, not a universal prompt improvement. The S−A
comparison also includes the search fix.

**B — purchase interception.** Prevents premature purchases with incomplete options,
unknown price or an explicit numeric budget violation. In S it blocked 34 attempts
on 27 tasks; eight later achieved gold purchases, and all eight failed in S−B
(8491, 15991, 2528, 11000, 15638, 5049, 18372, 7826). Wrong purchases rose from 10 to
21 when B was disabled; repeat loops fell from 13 to 8. This is evidence of useful
program assistance with a recovery/loop tradeoff. It is not evidence that the model
independently verified every requirement, nor that B checks all semantics.

**C — executable labels and recovery.** Preserves exact option labels, resolves only
unambiguous whitespace variants and supplies page-specific recovery guidance.
S−C retains B and its dedicated rejection feedback. Unique whitespace repair fired
three times on task 7506; exact-label whitespace differed on five tasks. These
observations explain a concrete interface mechanism, but do not explain the whole
10-success difference, which also includes feedback and the search-fix confound.
The fixed veRL schema was used; dynamic tool filtering was not evaluated or included.

**D — observation capacity.** Keeps more of each returned observation. On the saved
S paths, 241 of 242 search observations (199 tasks) exceeded the old 1536-token
budget. Product observations peaked at 1049 tokens, below the old 4096 limit;
information subpages peaked at 623, below 768. The observed mechanism is mainly
search-result/title retention. The S−D difference does not establish a general
benefit from larger detail budgets, and includes the search-fix confound.

**E — history management.** Compacts accumulated context and reserves generation
space. S recorded 16 compactions on only three tasks (7506, 10176, 13989), none of
which succeeded. S−E scored one success higher. No success-rate benefit is demonstrated
here, and this is not proof of equivalence or of no utility on longer tasks. Disabled
compaction's nominal input budget is not a hard cap; reserve changes are not injected
per-turn output limits. Keep E optional for further matched evaluation.

**F — catalog integrity.** Supplies reviewed source fields and the search index built
from them. The initial S−F had two infrastructure failures rather than valid model
failures. After replacing only those two, the mixed 200-task view scores 128, two
below S. This is limited evidence about data consistency, not justification to derive
actor requirements from hidden target answers. Repair provenance and catalog/index
pairing are required. The concurrency repair is independent of F and worked on both
catalogs; the initial 63% must not be reported as the final clean F result.

**G — environment judgement.** Reviews purchase constraints and candidate eligibility.
It affects which candidates are known acceptable and when abstention is available,
although strict purchase scoring uses the original result. S−G gains five and loses
four, scoring one higher than S. This does not demonstrate a success-rate gain from G.
Preserve it as an explicit policy switch, and distinguish reviewed diagnostics from
original success. A reviewed `reward_unverifiable` termination can coexist with an
original gold result; counting termination tags would misstate the experiment.

## Search repair and F provenance

The first S−F run failed on 21841 (`A800 NVLink 80G`, `float(None)`) and 11543
(`藏海传 眼部护理 眼霜`, SQLite “bad parameter or other API misuse”). Concurrent access
to a shared SQLite connection reproduced errors and silent ranking differences on
**both** original and reviewed indexes. The sampled pre-fix stress tests observed
6/800 ranking mismatches on each index. Serializing search/contains/close yielded
800/800 exact serial-order/score matches per index, 1600 total, without changing SQL,
query tokens, BM25 weights, data or tie ordering.

Only 21841 and 11543 were rerun for F; both succeeded. Their worker count and validation
batch size were two to avoid veRL padding duplicate trajectories; model loading
still used four GPUs. The other 198 trajectories and all original 200 raw files
remained byte-identical. A separate merged view retains per-task provenance. It is
**not** a full post-fix F rerun. A/C/D/E each contribute 200 new trajectories; together
with F's two this is 802, with no new invalid trajectories.

S, B, G and original F predate synchronization. A, C, D, E and F's two replacement
trajectories use it. Even pre-fix trajectories without exceptions may have experienced
ranking drift. A repaired S was not run, so the new-versus-S differences cannot be
attributed entirely to the disabled component. Single greedy trials also do not
establish statistical stability, additive contributions, or an optimal combination.

## Artifact identity and availability

The results were computed from completed frozen trajectories and terminal records;
raw task/catalog/model artifacts are not bundled in this PR. These SHA-256 identities
allow the artifact owner to check the evaluated inputs:

| Input | SHA-256 |
|---|---|
| Stage-C model | `3fbeff7f506dcad50c8f74ba5c5af1465d7d1d6ce020895c982872073d7e70f5` |
| Fixed task list | `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5` |
| Original products | `57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f` |
| Reviewed products | `9ad03b7dddef2e92f99227bbf0a17439e7eabd21722f10153cd0516e49f0702e` |

Hashes identify inputs; they do not make unavailable artifacts publicly reproducible.
The repository's synthetic tests cover component boundaries, original/reviewed score
selection, conservative option handling, catalog/index mismatch rejection, reviewed
policy fixtures and serial-versus-concurrent search. They do not reproduce the GPU
scores. This contribution adds no new model-evaluation claim and no RL improvement claim.
