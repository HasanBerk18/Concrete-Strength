# Experiment summary

## Data

```json
{
  "n_raw": 1030,
  "missing_values": 0,
  "exact_duplicates_removed": 25,
  "n_after_cleaning": 1005,
  "iqr_outliers_kept": {
    "cement": 0,
    "slag": 2,
    "fly_ash": 0,
    "water": 15,
    "superplasticizer": 10,
    "coarse_agg": 0,
    "fine_agg": 5,
    "age": 59,
    "strength": 8
  },
  "n_unique_mixes": 428,
  "mixes_tested_at_multiple_ages": 182,
  "rows_in_multi_age_mixes": 759,
  "replicate_rows_same_mix_and_age": 9
}
```

## Mix-grouped split

| subset | rows | row_share | mixes | strength_mean | strength_std | n_1-7 d | n_14-28 d | n_56-100 d | n_120-365 d |
|---|---|---|---|---|---|---|---|---|---|
| train | 705 | 0.701 | 298 | 35.107 | 15.800 | 178 | 336 | 145 | 46 |
| val | 162 | 0.161 | 65 | 35.163 | 16.783 | 39 | 76 | 37 | 10 |
| test | 138 | 0.137 | 65 | 36.086 | 18.133 | 36 | 69 | 27 | 6 |

## Test-set results (single evaluation)

| model | RMSE | MAE | R2 | train_RMSE | val_RMSE | size | size_unit | train_time_s |
|---|---|---|---|---|---|---|---|---|
| LinearRegression | 8.362 | 5.934 | 0.786 | 6.807 | 6.781 | 11 | coefficients | 0.001 |
| RandomForest | 6.523 | 4.472 | 0.870 | 1.776 | 5.442 | 263914 | tree nodes | 0.256 |
| SVR | 6.702 | 4.587 | 0.862 | 3.904 | 5.435 | 417 | support vectors | 0.021 |
| MLP-A | 7.554 | 5.308 | 0.825 | 4.505 | 4.843 | 385 | weights | 4.507 |
| MLP-B | 6.683 | 4.546 | 0.863 | 3.233 | 4.604 | 11777 | weights | 2.755 |
| MLP-C | 6.893 | 4.682 | 0.854 | 3.621 | 4.514 | 11777 | weights | 4.712 |

## Best hyperparameters (chosen on validation RMSE)

- **LinearRegression**: `{}`
- **RandomForest**: `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.5}`
- **SVR**: `{'C': 100, 'gamma': 0.05, 'epsilon': 2.0}`
- **MLP-A**: `{'lr': 0.003, 'batch_size': 16}`
- **MLP-B**: `{'lr': 0.001, 'batch_size': 32}`
- **MLP-C**: `{'lr': 0.001, 'batch_size': 32, 'dropout': 0.1, 'l2': 0.001}`

## Test RMSE by curing-age group (MPa)

| model | 1-7 d | 120-365 d | 14-28 d | 56-100 d |
|---|---|---|---|---|
| LinearRegression | 6.235 | 10.444 | 9.469 | 7.224 |
| RandomForest | 4.479 | 1.820 | 7.690 | 6.234 |
| SVR | 4.403 | 2.046 | 8.202 | 5.558 |
| MLP-A | 4.689 | 6.612 | 8.387 | 8.535 |
| MLP-B | 4.864 | 2.358 | 7.510 | 7.166 |
| MLP-C | 4.824 | 4.136 | 7.843 | 7.130 |

## Extrapolation: test RMSE inside vs. outside training input range (10 rows outside)

| model | inside | outside |
|---|---|---|
| LinearRegression | 8.049 | 11.645 |
| RandomForest | 5.861 | 12.140 |
| SVR | 6.083 | 12.090 |
| MLP-A | 6.384 | 16.301 |
| MLP-B | 5.483 | 15.216 |
| MLP-C | 5.646 | 15.739 |

## Hardest test samples (largest mean |error| over all models)

| mix_id | age | w_c | actual | mean_abs_err | outside_train_range |
|---|---|---|---|---|---|
| 76 | 28 | 0.460 | 81.751 | 24.511 | coarse_agg |
| 78 | 28 | 0.366 | 78.800 | 23.381 | - |
| 90 | 28 | 0.950 | 49.249 | 19.151 | coarse_agg |
| 79 | 28 | 0.314 | 41.369 | 17.330 | - |
| 77 | 28 | 0.386 | 64.018 | 17.259 | - |
| 0 | 28 | 0.300 | 79.986 | 16.665 | - |
| 30 | 91 | 0.406 | 56.496 | 15.469 | coarse_agg |
| 198 | 7 | 0.626 | 30.957 | 14.882 | - |
| 126 | 28 | 0.396 | 72.099 | 14.418 | - |
| 30 | 56 | 0.406 | 54.896 | 13.944 | coarse_agg |

## Repeated group splits (mean / std over seeds)

| model | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std |
|---|---|---|---|---|---|---|
| LinearRegression | 7.327 | 0.626 | 5.612 | 0.296 | 0.782 | 0.031 |
| RandomForest | 5.305 | 0.856 | 3.885 | 0.530 | 0.887 | 0.019 |
| SVR | 5.712 | 1.055 | 4.099 | 0.650 | 0.867 | 0.041 |
| MLP-A | 6.501 | 0.734 | 4.825 | 0.441 | 0.829 | 0.026 |
| MLP-B | 5.793 | 1.015 | 4.030 | 0.609 | 0.862 | 0.044 |
| MLP-C | 5.972 | 1.107 | 4.292 | 0.644 | 0.854 | 0.046 |
