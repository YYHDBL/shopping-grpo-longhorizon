# Reward v3

Reward v3 is a deterministic terminal reward for constraint-aware shopping. It
scores what the agent actually selected using environment evidence; it does not
ask another language model to judge the answer.

## 1. Compile task requirements

Before rollout, the task and target-product metadata are converted into fixed
features:

- category and upper price bound;
- explicitly mentioned brand aliases;
- model identifiers shared by the instruction and target metadata;
- required core functions;
- required option values such as color, size, capacity or bundle.

This compilation is frozen before the policy acts. The policy cannot change the
requirements it will later be scored against.

## 2. Apply hard gates

A purchase must pass both:

- **Category:** the selected product belongs to the required category.
- **Budget:** the resolved price of the selected variant is at or below the
  user's upper bound.

A failed hard gate produces `wrong_purchase` with reward `-0.85`. If a hard
gate cannot be verified from environment evidence, the result is
`reward_unverifiable`, reward `0.0`, and `reward_valid=false`. This zero is not
treated as a successful neutral outcome.

## 3. Score preferences

Four soft dimensions are scored among only the dimensions active for that task:

| Dimension | Weight |
|---|---:|
| Brand | 0.35 |
| Model | 0.25 |
| Core functions | 0.25 |
| Key options | 0.15 |

For active dimensions, the match score is:

```text
S = Σ(weight_i × score_i) / Σ(active weight_i)
```

Evidence coverage is aggregated with the same active weights. Full satisfaction
requires both match score and coverage to equal 1.0.

## 4. Map the terminal state to reward

| Outcome | Reward |
|---|---:|
| Exact target ASIN, all requirements satisfied | `1.00` |
| Different ASIN, all requirements satisfied | `0.55` |
| Partial alternative | `min(0.25, -0.30 + 0.55 × S)` |
| Graceful stop after sufficient search | `-0.15` |
| Stop before sufficient search | `-0.35` |
| Maximum 35 steps | `-0.50` |
| Repeated/no-progress loop | `-0.65` |
| Wrong category or over budget | `-0.85` |
| Required evidence unavailable | `0.00`, invalid |

The exact target is called `gold_purchase`; a different item that passes the
same hard gates and fully satisfies every active preference is
`valid_alternative_purchase`. This prevents the reward from treating a single
catalog identifier as the only correct solution.

## 5. Responsible abstention and termination

Stopping is considered graceful only after the agent has inspected at least two
effective result sets, opened at least two candidates and found no known
acceptable candidate. A candidate is “known acceptable” when both hard gates
pass, preference match is at least 0.70 and evidence coverage is at least 0.75.

The environment also ends a rollout after:

- two consecutive exact repeats;
- four consecutive actions with no new runtime evidence;
- 35 total steps.

Search results, newly opened products, detail subpages, selected options and
constraint checks count as evidence. Evidence credits are bounded so an agent
cannot farm progress indefinitely.

## Source of truth

The constants are frozen in
[`environments/ShopSimulator/shop_env/configs/environment.json`](../environments/ShopSimulator/shop_env/configs/environment.json).
The implementation is
[`reward.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py),
with termination logic in
[`termination.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py).

## Variant price caveat

Multiple effective option axes do not prove that every combination is
purchasable. A listing may bind options into complete variants, such as
`white + small` and `black + large`, while exposing color and size as separate
fields. When `variant_combinations` or equivalent SKU linkage is missing, the
resolver cannot safely infer the price of an arbitrary cross-combination and
returns `reward_unverifiable`.

Such a result records missing price evidence, not proof that every model must
fail the task. Do not remove an evaluation task solely from a model trajectory
with this result. Reassess the product using complete SKU option, linkage and
price evidence first.
