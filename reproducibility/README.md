# Reproducibility materials

This directory records the materials needed to inspect and rerun the released
downstream DGraph pipeline without depending on an external LLM API.

## Scope

During development, an LLM was used to assist with feature-engineering code.
The release reference is the final reviewed source code, the fixed DGraph
feature order in [`dgraphfin_final_features.json`](dgraphfin_final_features.json),
and the definitions below. The historical API interaction was not
programmatically logged; therefore, this repository does not claim that a new
API invocation will regenerate identical code or results.

The relevant released code is:

- feature extraction: [`../code_cross/`](../code_cross/);
- final feature order and downstream training:
  [`../code_cross/train.py`](../code_cross/train.py); and
- downstream testing:
  [`../code_cross/test.py`](../code_cross/test.py).

This page describes the released code snapshot. The experimental protocol and
data split for reported paper results are stated in the paper.

## Available prompt material

### Feature-engineering code

The retained task instruction was to conduct feature engineering on a large,
dynamic financial graph and output a node-feature table in `.pkl` format. The
DGraph input was described as an `.npz` file containing `x`, `y`,
`edge_index`, `edge_type`, `edge_timestamp`, `train_mask`, and `test_mask`.
The instruction requested `float32` features, modular feature-engineering
scripts with explicit input/output arguments, bounded neighbor sampling (for
example, 50 one-hop and 500 two-hop neighbors), and `DataFrame` outputs saved
as `.pkl` files.

The retained material also described DGraph as a large dynamic financial graph,
with background-node labels 2 and 3 and a highly imbalanced fraud-detection
target. It is preserved here as development documentation; the final reviewed
code and fixed feature specification are authoritative for the released
pipeline.

### Cross-feature proposals

The retained cross-feature instruction was:

> Now I have some input features and need you to help me with
> cross-combination. The maximum number of new combined features should not
> exceed 30. I will provide an appendix `.txt` file containing the statistics
> of the current top features. The current input features are:
> `[fea_1, fea_2, ...]`

The instantiated feature list and the referenced statistics file were not
retained. The actual cross features used by the released DGraph code are frozen
below.

## Frozen DGraph cross features

The function `create_cross_features_dgraphfin` in `code_cross/train.py`
conditionally creates each feature below only when its named inputs are
available. Here, `epsilon = 1e-6`.

| # | Feature | Definition |
|---:|---|---|
| 1 | `active_window_span` | `last_active_out - undirected_1hop_timestamp_min` |
| 2 | `risk_transaction_density` | `undirected_1hop_f16_max / (directed_1hop_out_timestamp_range + epsilon)` |
| 3 | `recent_burst_risk` | `(1 / (last_active_out + 1)) * undirected_1hop_f16_max` |
| 4 | `timing_anomaly_score` | `directed_1hop_out_timestamp_range * directed_1hop_out_interval_std` |
| 5 | `core_feature_interaction` | `fea_x_2 * fea_x_11 / (abs(fea_x_6) + epsilon)` |
| 6 | `dormant_high_value_risk` | `inactive_days_out * undirected_1hop_f15_max` |
| 7 | `out_timestamp_concentration` | `1 / (directed_1hop_out_timestamp_max - directed_1hop_out_timestamp_min + 1)` |
| 8 | `edge_type_discrepancy` | `fea_out_edge_type_max - fea_out_edge_type_mean` |
| 9 | `first_interaction_gap` | `directed_1hop_out_timestamp_min - undirected_1hop_timestamp_min` |
| 10 | `weighted_x_features` | `0.35 * fea_x_2 + 0.25 * fea_x_1 + 0.20 * fea_x_0 + 0.15 * fea_x_15 + 0.05 * fea_x_8` |
| 11 | `transaction_stability` | `1 / (directed_1hop_out_timestamp_range * directed_1hop_out_interval_std + 1)` |
| 12 | `high_freq_high_value` | `undirected_1hop_f16_max / (log1p(directed_1hop_out_timestamp_range) + 1)` |
| 13 | `activity_inconsistency` | `last_active_out * inactive_days_out` |
| 14 | `early_wide_spread_risk` | `undirected_1hop_timestamp_min * log1p(directed_1hop_out_timestamp_range)` |
| 15 | `multi_risk_composite` | `fea_x_2 * (1 / (last_active_out + 1)) * log1p(undirected_1hop_f16_max)` |
| 16 | `core_time_interaction` | `(last_active_out - undirected_1hop_timestamp_min) * (1 / (directed_1hop_out_timestamp_max - undirected_1hop_timestamp_min + 1))` |

## Downstream training snapshot

`code_cross/train.py` uses a five-fold shuffled `StratifiedKFold` with seed
`2022`. The `CatBoostClassifier` configuration is: CPU execution, 2,500
iterations, learning rate 0.06, depth 5, `l2_leaf_reg=3`, `border_count=254`,
Bernoulli bootstrap with `subsample=0.85`, `sampling_frequency=PerTree`,
`grow_policy=SymmetricTree`, `random_strength=1.5`, balanced class weights,
Logloss objective, AUC evaluation, and early stopping after 200 rounds.
