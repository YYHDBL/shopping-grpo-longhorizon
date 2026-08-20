# SFT Data Mixing Diagnosis

> Date: 2026-08-20. This note documents a small diagnostic finding about mixing
> independently collected SFT trajectories for a tool-using shopping agent.

## Summary

Merging successful trajectories from multiple DeepSeek-v4-Flash collection runs
can still hurt SFT performance when the merged sources have different assistant
output styles.

In our diagnosis, the additional merged trajectories were valid shopping
trajectories, but they were much more verbose than the original SFT data. Under
full assistant loss, these long natural-language rationale tokens became a large
part of the supervised target. This likely shifted the student model toward
explaining, reconsidering, or producing assistant-final style text instead of
reliably emitting executable tool calls.

The key lesson is that for tool-use SFT, trajectory success is not the only data
quality criterion. The imitation target also needs to be format-stable,
action-dense, and close to the desired inference-time behavior.

## Observed Results

We compared three 4B SFT runs on the same 200-task held-out evaluation set:

| Variant | Held-out strict success |
|---|---:|
| Original SFT data | 142 / 200 = 71.0% |
| Mixed-source SFT data with full assistant loss | 112 / 200 = 56.0% |
| Same mixed-source data with tool-call-only loss | 135 / 200 = 67.5% |

The full-loss mixed-source run dropped substantially relative to the original
SFT run. Training on only the tool-call spans recovered much of the lost
performance, suggesting that the merged trajectories' natural-language
assistant content was a major factor.

## Source-Level Difference

The mixed dataset contained two broad sources:

- the original/current SFT source;
- an additional externally collected merged source.

The merged source was much more verbose:

| Source within mixed data | Row share | Assistant content character share | Tool-call share |
|---|---:|---:|---:|
| Original/current source | 79.1% | 45.1% | 74.6% |
| Additional merged source | 20.9% | 54.9% | 25.4% |

Although the merged source was only about one fifth of the rows, it contributed
more than half of the assistant natural-language characters. This means it had a
disproportionately large effect when full assistant loss was used.

More detailed differences:

| Metric | Original/current source | Additional merged source |
|---|---:|---:|
| Mean assistant content chars / row | 264.9 | 1,218.8 |
| Median assistant content chars / row | 255.5 | 854 |
| P90 assistant content chars / row | 468 | 2,214 |
| Mean non-empty assistant turns | 3.98 | 8.37 |
| Mean assistant turns over 100 chars | 0.61 | 3.88 |
| Mean tool calls | 7.39 | 9.51 |
| English-like non-empty assistant turns | 12.6% | 29.0% |

Qualitatively, the merged trajectories often contained longer reasoning text,
more "let me verify / reconsider" style narration, more English or mixed-language
assistant content, and more intermediate checking before purchase.

## Interpretation

The additional merged data does not appear to be simply "bad" data. It can
contain successful and reasonable shopping behavior. The problem is that its
assistant text distribution is different from the original SFT distribution and
from the desired inference-time behavior of a tool-using agent.

With full assistant loss, the model is trained to reproduce both:

- the executable tool calls;
- the natural-language rationale around those tool calls.

If one source is much more verbose, it can dominate the assistant-token loss even
when it is a minority of examples. For an agent whose main requirement is stable
tool use, that can push the model toward extra text, less stable action format,
and occasional assistant-final responses.

The tool-call-only ablation supports this explanation: keeping the same mixed
data but excluding the natural-language rationale from the loss recovered a large
part of the performance drop.

## Practical Takeaways

When merging independently collected tool-use trajectories, it is useful to
audit not only final success but also imitation-target quality:

- assistant content length;
- ratio of tool-call tokens to natural-language rationale;
- language/style drift between sources;
- number of non-empty assistant turns;
- trajectory length and repeated reconsideration;
- source-level contribution to supervised tokens, not only row count.

For mixed-source SFT data, lower-risk options include:

1. training only on tool-call spans;
2. trimming or removing assistant rationale text;
3. source-balancing by assistant-token count rather than row count;
4. preferring successful trajectories that are concise and action-dense.

The broader conclusion is that successful teacher trajectories can still be
distributionally mismatched for SFT. For tool-using agents, data merging should
preserve both outcome quality and output-format consistency.

