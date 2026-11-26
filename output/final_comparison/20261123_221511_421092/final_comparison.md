# 最终结果对比

分类来源：`output/classification_comparison/20261123_220637_250154`

回归来源：`output/regression_comparison/20261123_220645_862295`

## 分类：A 组与 B 组

| scenario | n_test | roc_auc | ap | accuracy | balanced_accuracy | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| A 组 · 6 特征 + XGBoost | 449 | 0.542628 | 0.463050 | 0.530067 | 0.530738 | 0.486618 |
| B 组 · 11 特征 + XGBoost | 449 | 0.557589 | 0.460206 | 0.552339 | 0.568192 | 0.552339 |

## 回归：XGBoost 与岭回归

| scenario | n_test | mae_bp | rmse_bp | r2 | constant_rmse_bp | delta_rmse_bp |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 · 6 特征 + XGBoost | 752 | 62.355094 | 92.674628 | 0.004790 | 92.995789 | -0.321161 |
| 最终 · 6 特征 + 岭回归 | 752 | 62.340638 | 92.496864 | 0.008604 | 92.995789 | -0.498925 |

## 实际样本划分

分类：

| split | samples | first_signal | last_signal | first_entry | last_settlement |
| --- | --- | --- | --- | --- | --- |
| train | 3593 | 2009-01-02 | 2023-04-12 | 2009-01-05 | 2023-04-14 |
| val | 448 | 2023-04-14 | 2025-01-27 | 2023-04-17 | 2025-01-29 |
| test | 449 | 2025-01-29 | 2026-11-10 | 2025-01-30 | 2026-11-12 |

回归：

| split | samples | first_signal | last_signal | first_entry | last_settlement |
| --- | --- | --- | --- | --- | --- |
| train | 6020 | 1996-12-12 | 2020-11-11 | 1996-12-13 | 2020-11-13 |
| val | 751 | 2020-11-13 | 2023-11-08 | 2020-11-16 | 2023-11-10 |
| test | 752 | 2023-11-10 | 2026-11-10 | 2023-11-13 | 2026-11-12 |

## 指标差值（final − baseline）

| task | metric | baseline | final | delta_final_minus_baseline | better_direction |
| --- | --- | --- | --- | --- | --- |
| classification | roc_auc | 0.542628 | 0.557589 | 0.014961 | higher |
| classification | ap | 0.463050 | 0.460206 | -0.002845 | higher |
| classification | accuracy | 0.530067 | 0.552339 | 0.022272 | higher |
| classification | balanced_accuracy | 0.530738 | 0.568192 | 0.037454 | higher |
| classification | f1 | 0.486618 | 0.552339 | 0.065721 | higher |
| regression | mae_bp | 62.355094 | 62.340638 | -0.014456 | lower |
| regression | rmse_bp | 92.674628 | 92.496864 | -0.177764 | lower |
| regression | r2 | 0.004790 | 0.008604 | 0.003814 | higher |

回归收益率以基点展示，1% = 100 基点。delta_rmse_bp 为模型减训练均值常数基准，负值表示改善。

分类新增跨市场特征只改变输入；回归比较改变模型。14 特征分析表没有自动成为分类 B 组输入。

这些是单次历史时序留出结果，数据此前已参与方案探索；小幅差值不构成统计显著性证据，也不保证未来有效。
