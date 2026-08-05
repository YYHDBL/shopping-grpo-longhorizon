# Feed frozen evaluation

- Schema: `feed-frozen-evaluation-v1`
- Split: `test`
- Paired policies: `true`
- Scoring: deterministic simulator metrics; no LLM judge

| Policy | long_term_return | qualified_purchase_rate | correct_no_recommend_rate | interventions_per_100 | return_rate | irrelevant_recommendation_rate | repeat_exposure_rate | grounded_recommendation_rate | unsupported_claim_rate | mean_dwell_seconds | skip_rate | terminal_satisfaction | complementary_bundle_precision | net_revenue | terminal_fatigue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popular | -13.870875 | 0.000000 | 0.000000 | 100.000000 | 0.000000 | 0.791667 | 0.625000 | 1.000000 | 0.000000 | 12.390125 | 0.250000 | 0.588560 | 0.000000 | 59.000000 | 1.000000 |
| Random | -6.618850 | 0.000000 | 0.600000 | 54.166667 | 0.000000 | 0.923077 | 0.307692 | 1.000000 | 0.000000 | 15.451458 | 0.083333 | 0.676897 | 0.000000 | 198.293810 | 0.892500 |
| Rule | 0.091635 | 0.666667 | 0.000000 | 29.166667 | 0.333333 | 0.428571 | 0.285714 | 1.000000 | 0.000000 | 18.556375 | 0.000000 | 0.578935 | 0.000000 | 96.000000 | 0.190000 |
| Similarity | -3.830906 | 0.666667 | 0.000000 | 100.000000 | 0.333333 | 0.166667 | 0.041667 | 1.000000 | 0.000000 | 13.030375 | 0.208333 | 0.590455 | 0.000000 | 74.000000 | 1.000000 |
| Teacher | -0.753654 | 0.500000 | 0.666667 | 45.833333 | 0.500000 | 0.545455 | 0.090909 | 1.000000 | 0.000000 | 17.858208 | 0.000000 | 0.545055 | 1.000000 | 96.000000 | 0.310000 |

> Reward is not collapsed with experience or safety metrics; inspect every column.
