# HAR EDA Summary Report

## Dataset Overview
| Metric | Value |
|--------|-------|
| Total timesteps | 3,306,000 |
| Total sequences (CSV files) | 11020 |
| Number of features | 6 |
| NaN values present | False |

## Class Distribution
| Label | Name | Count | Pct |
|-------|------|-------|-----|
| 0 | L0 | 4643 | 42.13% |
| 1 | L1 (Running) | 4695 | 42.60% |
| 2 | L2 (Stairs Up) | 358 | 3.25% |
| 3 | L3 | 656 | 5.95% |
| 4 | L4 | 142 | 1.29% |
| 5 | L5 | 526 | 4.77% |

## Feature Global Statistics
|        |       mean |       std |      min |     max |
|:-------|-----------:|----------:|---------:|--------:|
| mean_x | -0.144502  | 0.619214  | -2.04313 | 1.41968 |
| mean_y |  0.0114329 | 0.465017  | -2.76188 | 4.13459 |
| mean_z |  0.195908  | 0.575272  | -1.71906 | 1.22565 |
| std_x  |  0.051437  | 0.10677   |  0       | 4.10899 |
| std_y  |  0.0440868 | 0.100924  |  0       | 3.71988 |
| std_z  |  0.0470486 | 0.0961562 |  0       | 3.7538  |

## Files Generated
- `plots/01_label_distribution.png`
- `plots/02_deep_eda_grand_unified.png`
- `plots/03_feature_space_L1_vs_L2.png`
- `plots/04_spatial_geometry_L1_vs_L2.png`
- `plots/05_global_acf_L1_vs_L2.png`
- `reports/quantitative_mining_report.txt`
- `reports/feature_importance.csv`
- `reports/feature_correlation_matrix.csv`
- `reports/global_feature_stats.csv`
- `reports/class_feature_stats.csv`
- `reports/missing_value_report.csv`
- `reports/full_data_summary.md`