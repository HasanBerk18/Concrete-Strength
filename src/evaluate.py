"""Metrics, grouped error breakdowns and small table helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y, p) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
    }


def group_breakdown(groups: pd.Series, y, preds: dict) -> pd.DataFrame:
    """RMSE / MAE / bias per group. Bias = mean(predicted - actual).

    R² is omitted on purpose: within a narrow group the target variance is
    small, so R² is not comparable across groups.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    rows = []
    for name, p in preds.items():
        for g in groups.cat.categories:
            m = (groups == g).to_numpy()
            if not m.any():
                continue
            err = p[m] - y[m]
            rows.append({
                "model": name, "group": g, "n": int(m.sum()),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MAE": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
            })
    return pd.DataFrame(rows)


def worst_cases(test_df: pd.DataFrame, train_df: pd.DataFrame, y, preds: dict,
                features, top: int = 15) -> pd.DataFrame:
    """Test samples with the largest mean absolute error across all models.

    `outside_train_range` lists every input whose value lies outside the
    training min-max range, i.e. where the models must extrapolate.
    """
    lo, hi = train_df[features].min(), train_df[features].max()
    out = test_df[["mix_id", *features, "age"]].reset_index(drop=True).copy()
    out["actual"] = y
    abs_err = pd.DataFrame({f"abs_err_{n}": np.abs(p - y) for n, p in preds.items()})
    out = pd.concat([out, abs_err], axis=1)
    out["mean_abs_err"] = abs_err.mean(axis=1)
    outside = (test_df[features].reset_index(drop=True) < lo) | (test_df[features].reset_index(drop=True) > hi)
    out["outside_train_range"] = outside.apply(lambda r: ", ".join(r.index[r]) or "-", axis=1)
    return out.sort_values("mean_abs_err", ascending=False).head(top).reset_index(drop=True)


def extrapolation_effect(test_df, train_df, y, preds: dict, features) -> pd.DataFrame:
    """Per-model RMSE for test rows inside vs. outside the training input range."""
    lo, hi = train_df[features].min(), train_df[features].max()
    X = test_df[features].reset_index(drop=True)
    outside = ((X < lo) | (X > hi)).any(axis=1).to_numpy()
    rows = []
    for name, p in preds.items():
        for flag, label in ((False, "inside"), (True, "outside")):
            m = outside == flag
            if m.any():
                rows.append({"model": name, "train_range": label, "n": int(m.sum()),
                             "RMSE": float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
