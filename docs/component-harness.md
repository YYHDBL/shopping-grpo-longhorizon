# Optional component harness

This opt-in veRL adapter separates prompt, purchase checks, interface assistance,
observation budgets, context handling, catalog quality and environment judgement.
The baseline configuration and entrypoints retain their defaults. Description,
Attributes and Reviews still require explicit tools. No automatic detail fields,
dynamic tool-schema filtering, new failure penalty or training launcher is added.
See [the evaluation evidence](component-evaluation.md) before interpreting the switches.

## Component boundaries

| Component | Enabled behavior | Disabled behavior | Implementation |
|---|---|---|---|
| A: prompt | Explicit information collection, option selection and pre-purchase verification instructions | Existing system prompt | `configs/components/system_prompt.txt` |
| B: purchase prerequisites | Reject incomplete options, unknown price or a violated explicit numeric budget before `buy_now` | Existing action guard only | `environment/purchase_support.py`, `budget.py` |
| C: interface assistance | Preserve executable label whitespace; resolve current-page option IDs or unique whitespace-only matches; actionable recovery feedback | Original renderer and guard feedback | `environment/interface_support.py` |
| D: observation budgets | Search/detail/generic budgets all 8192 tokens | 1536/4096/768 tokens | `configs/components/observation.json` |
| E: context handling | Compaction enabled, input target 22016, generation reserve 2048 | Compaction disabled, configured target 16384, reserve 512 | `configs/components/context.json` |
| F: catalog quality | Explicitly reviewed catalog and its matching index | Original catalog and its matching index | `environment/task_data_quality.py`, profile pair validation |
| G: environment judgement | Reviewed purchase **and candidate eligibility** functions | Original functions | `engine/reviewed_reward.py`, `scripts/serve_reviewed_environment.py` |

B uses only the public task and current observation. It is not a semantic proof
that the product meets every requirement. Unknown or vague budget language must
not become an invented numeric constraint. C never chooses a product or option by
meaning; ambiguous matches remain unresolved. B and C are independent flags in
each tool configuration. Their shared action guard is still the upstream guard.
The tool schemas are fixed, as in the evaluated veRL path; current-page IDs are
accepted by the resolver but are not injected through a dynamic schema.

D changes one observation's projection, while E changes accumulated history.
With E disabled, 16384 is not an enforced hard cap: the effective context hard
limit is 24576 − 512 − 512 = 23552. A generation reserve is not a per-turn
`max_tokens` sampling argument. The evaluated sampling parameters did not supply
such a cap.

F is a data intervention, not a model improvement. The data-quality module audits
and applies explicit repairs; it does not ship the evaluated catalog or infer
missing requirements from hidden answers. Tasks 14671 and 2717 remain in the
200-task denominator, with incomplete requirements flagged for review.
Selecting F requires supplying the reviewed catalog; disabling it requires the
original catalog. The builder checks catalog/index SHA-256 consistency, not the
correctness of a human review or whether a path contains the intended snapshot.

G changes both `evaluate_purchase` and `evaluate_candidate_eligibility`. Eligibility
feeds `known_valid_asins`, progress tracking and `finish_without_purchase`; it is
not just a reporting label. Reviewed purchase results preserve original Reward v3
under `evidence.legacy_result`. The adapter selects this original score for
purchase reward and strict-success fields. With G disabled it accepts direct
original Reward v3. Reviewed auxiliary dimensions remain diagnostics, and existing
nonpurchase terminal rewards remain unchanged.

## Compose a profile without running an experiment

Install the project's veRL environment (including PyYAML), then run from the
repository root. Paths below are user-supplied inputs, not bundled datasets:

```sh
python scripts/build_component_profile.py \
  --products /path/to/reviewed-products.json \
  --index /path/to/reviewed-index.sqlite \
  --output /path/to/new-profile
```

Use `--disable B`, for example, to compose S−B. Use `--disable F` with the
**original** catalog and original matching index for S−F. Multiple switches are
accepted for inspection; the reported experiments disabled one component at a time.
The output directory must be new. The command only writes four configuration
files; it starts no service, model or training process.

- `agent_loop.yaml`: JSON-compatible YAML selecting the optional agent and A/D/E.
- `tools.json`: unchanged tool schemas, optional tool class and independent B/C flags.
- `service.json`: explicit service argument vector and environment, including the
  matching `SHOP_SEARCH_INDEX`; G selects `--reviewed-reward`.
- `protocol.json`: component flags, data paths/hash and fixed observation/scoring policy.

When independently authorized to run an evaluation, start the service using the
argument vector and environment in `service.json`, and pass these existing veRL
overrides to the evaluation configuration:

```text
actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/new-profile/agent_loop.yaml
actor_rollout_ref.rollout.multi_turn.tool_config_path=/path/to/new-profile/tools.json
```

Set `SHOPSIM_BASE_URL` to that service. The service forces automatic details off.
Optional public option-image evidence must be supplied explicitly with
`--option-evidence`; it affects evidence judgement, not automatic observations.
`trajectory_directory` may be set in the generated agent configuration to record
public environment events, raw calls, response tokens and loss masks. These are
local audit files, not model prompt additions. No bundled model runner or audit
hook is needed merely to compose a profile.

## Shared search fix

The SQLite searcher was shared by request threads. A per-searcher reentrant lock
now serializes `search`, `contains_asin` and `close`, without changing SQL, ranking
weights or data. It can reduce concurrent throughput; correctness comes first.
This is shared infrastructure, not component F. Synthetic concurrency tests check
exact hit order and score against serial requests. Historical stress-test evidence
and the resulting limits on component attribution are recorded separately.

## CPU validation

Run with the project's development/veRL dependencies and the environment package
on `PYTHONPATH` (no model or service is started):

```sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:.:environments/ShopSimulator/shop_env \
python -m pytest -q \
  tests/test_component_adapter.py tests/test_component_profile.py \
  tests/test_component_purchase.py tests/test_legacy_reward.py \
  tests/test_task_data_quality.py tests/test_reviewed_service.py \
  tests/test_action_validation.py tests/test_structured_observation.py \
  tests/test_context_window.py tests/test_verl_adapter.py \
  environments/ShopSimulator/shop_env/tests/test_search_concurrency.py \
  environments/ShopSimulator/shop_env/tests/test_public_option_evidence.py \
  environments/ShopSimulator/shop_env/tests/test_reviewed_reward.py \
  environments/ShopSimulator/shop_env/tests/test_reviewed_public_constraints.py \
  environments/ShopSimulator/shop_env/tests/test_review6_evidence.py \
  environments/ShopSimulator/shop_env/tests/test_remaining43_review.py
```

Observed result: **102 passed, 31 subtests passed, 5 failed**. All five failures are
in the unchanged `tests/test_verl_adapter.py`; the same five reproduce on a clean
checkout of base `449bba2686a6ff07327cb3ea77e5f0f91ebb4239` under the same environment:

- `test_repeated_guard_rejections_terminate_instead_of_looping_forever`
- `test_runtime_state_has_no_hidden_goal_fields`
- `test_terminal_observation_is_not_returned_to_the_model`
- `test_terminal_reward_components_are_validated_without_entering_tool_observation`
- `test_terminal_reward_keeps_unverifiable_separate_from_infrastructure`

The base-only adapter run had 14 passed and those 5 failed. This is not a fully
green upstream suite. Ruff check/format passed on all 27 changed Python files;
`git diff --check` passed. No GPU evaluation was run on this refactor.
