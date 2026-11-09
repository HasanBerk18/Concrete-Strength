"""Loading, cleaning, feature engineering, mix-grouped splitting and scaling."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import config as C


# ---------------------------------------------------------------- loading
def load_raw(path=C.RAW_XLS) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] != len(C.RAW_COLUMNS):
        raise ValueError(f"Expected {len(C.RAW_COLUMNS)} columns, got {df.shape[1]}")
    df.columns = C.RAW_COLUMNS
    return df


def iqr_outlier_counts(df: pd.DataFrame, cols) -> dict:
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = q3 - q1
    mask = (df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)
    return {c: int(v) for c, v in mask.sum().items()}


def clean(df: pd.DataFrame):
    """Drop missing and exact-duplicate rows; report (but keep) IQR outliers.

    Outliers are kept: they are physically plausible mixes (e.g. high
    superplasticizer, 365-day ages), not recording errors.
    """
    report = {"n_raw": len(df), "missing_values": int(df.isna().sum().sum())}
    df = df.dropna()
    dup = df.duplicated()
    report["exact_duplicates_removed"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)
    report["n_after_cleaning"] = len(df)
    report["iqr_outliers_kept"] = iqr_outlier_counts(df, C.RAW_COLUMNS)
    return df, report


# ---------------------------------------------------------------- features
def assign_mix_id(df: pd.DataFrame) -> pd.DataFrame:
    """Identical mix proportions (all 7 components) share one mix_id."""
    df = df.copy()
    key = df[C.MIX_COLUMNS].round(2)
    df["mix_id"] = key.groupby(C.MIX_COLUMNS, sort=False).ngroup()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["w_c"] = df["water"] / df["cement"]
    df["w_b"] = df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])
    df["log_age"] = np.log(df["age"])
    df["age_group"] = pd.cut(df["age"], bins=C.AGE_BINS, labels=C.AGE_LABELS)
    df["strength_group"] = pd.cut(df[C.TARGET], bins=C.STRENGTH_BINS, labels=C.STRENGTH_LABELS)
    return df


def prepare(path=C.RAW_XLS):
    df, report = clean(load_raw(path))
    df = add_features(assign_mix_id(df))
    ages_per_mix = df.groupby("mix_id")["age"].nunique()
    report["n_unique_mixes"] = int(df["mix_id"].nunique())
    report["mixes_tested_at_multiple_ages"] = int((ages_per_mix > 1).sum())
    report["rows_in_multi_age_mixes"] = int(df["mix_id"].map(ages_per_mix).gt(1).sum())
    report["replicate_rows_same_mix_and_age"] = int(df.duplicated(C.MIX_COLUMNS + ["age"]).sum())
    return df, report


# ---------------------------------------------------------------- splitting
@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def group_split(df: pd.DataFrame, seed: int, fractions=C.SPLIT_FRACTIONS) -> Split:
    """70/15/15 split by mix_id: a mix never appears in two subsets.

    Fractions apply to mixes (groups); row shares are therefore approximate.
    """
    f_train, f_val, f_test = fractions
    groups = df["mix_id"].to_numpy()
    idx = np.arange(len(df))

    outer = GroupShuffleSplit(n_splits=1, test_size=f_test, random_state=seed)
    trval, test = next(outer.split(idx, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=f_val / (f_train + f_val), random_state=seed)
    tr_rel, val_rel = next(inner.split(trval, groups=groups[trval]))
    split = Split(train=trval[tr_rel], val=trval[val_rel], test=test)

    g = {k: set(groups[getattr(split, k)]) for k in ("train", "val", "test")}
    assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]), "mix leakage"
    return split


def split_summary(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    rows = []
    for name in ("train", "val", "test"):
        part = df.iloc[getattr(split, name)]
        row = {
            "subset": name,
            "rows": len(part),
            "row_share": len(part) / len(df),
            "mixes": part["mix_id"].nunique(),
            "strength_mean": part[C.TARGET].mean(),
            "strength_std": part[C.TARGET].std(),
        }
        row.update({f"n_{g}": int(n) for g, n in part["age_group"].value_counts(sort=False).items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- arrays
@dataclass
class Arrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def make_arrays(df: pd.DataFrame, split: Split, features=C.FEATURES) -> Arrays:
    """Standardise inputs with a scaler fitted on the training subset only."""
    X = df[features].to_numpy(dtype=float)
    y = df[C.TARGET].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X[split.train])
    t = scaler.transform
    return Arrays(
        t(X[split.train]), y[split.train],
        t(X[split.val]), y[split.val],
        t(X[split.test]), y[split.test],
        scaler,
    )
