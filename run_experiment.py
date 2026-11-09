"""End-to-end experiment: data -> grouped split -> tuning -> one-time test evaluation -> figures.

Usage:
    python run_experiment.py            # full run
    python run_experiment.py --fast     # quick smoke test (tiny grids, few epochs)
    python run_experiment.py --no-repeats --skip-mlp-c
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import evaluate as E
from src import models as M
from src import plotting as P


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="tiny grids and few epochs (smoke test)")
    ap.add_argument("--no-repeats", action="store_true", help="skip the repeated-seed robustness study")
    ap.add_argument("--skip-mlp-c", action="store_true", help="omit the optional regularised MLP-C")
    return ap.parse_args()


def log(msg=""):
    print(msg, flush=True)


def main():
    args = parse_args()
    out = C.RESULTS_DIR
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    P.apply_style()
    t_start = time.perf_counter()
    model_names = [m for m in C.MODEL_ORDER if not (args.skip_mlp_c and m == "MLP-C")]

    # 1) Data ---------------------------------------------------------------
    log("== 1. Data preparation")
    df, report = D.prepare()
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    df.describe().T.to_csv(out / "descriptive_stats.csv")
    for k, v in report.items():
        log(f"   {k}: {v}")
    P.plot_target_distribution(df, figs / "eda_strength_distribution.png")
    P.plot_strength_vs_age(df, figs / "eda_strength_vs_age.png")
    P.plot_correlation(df, figs / "eda_correlation.png")

    # 2) Split --------------------------------------------------------------
    log(f"\n== 2. Mix-grouped 70/15/15 split (seed {C.MAIN_SEED})")
    split = D.group_split(df, C.MAIN_SEED)
    data = D.make_arrays(df, split)
    summary_split = D.split_summary(df, split)
    summary_split.to_csv(out / "split_summary.csv", index=False)
    log(summary_split[["subset", "rows", "row_share", "mixes", "strength_mean"]].to_string(index=False))

    # 3) Tuning on the validation set --------------------------------------
    log("\n== 3. Hyperparameter tuning (validation set only)")
    results, tuning = {}, {}
    for name in model_names:
        log(f"-- {name}")
        base = results["MLP-B"].params if name == "MLP-C" else None
        best, table = M.tune(name, C.MAIN_SEED, data, base_params=base, fast=args.fast, log=log)
        results[name], tuning[name] = best, table
        pd.DataFrame(table).sort_values("val_rmse").to_csv(out / f"tuning_{name}.csv", index=False)
        log(f"   best {best.params} -> val RMSE {best.val_rmse:.3f}")
    best_params = {n: r.params for n, r in results.items()}
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    # 4) Final evaluation: test set used exactly once -----------------------
    log("\n== 4. Final evaluation on the held-out test set")
    rows, test_preds = [], {}
    for name, res in results.items():
        p_tr, p_va, p_te = (res.model.predict(X) for X in (data.X_train, data.X_val, data.X_test))
        test_preds[name] = p_te
        m_tr, m_va, m_te = (E.regression_metrics(y, p) for y, p in
                            ((data.y_train, p_tr), (data.y_val, p_va), (data.y_test, p_te)))
        rows.append({
            "model": name,
            "RMSE": m_te["RMSE"], "MAE": m_te["MAE"], "R2": m_te["R2"],
            "train_RMSE": m_tr["RMSE"], "val_RMSE": m_va["RMSE"],
            "gap_test_minus_train": m_te["RMSE"] - m_tr["RMSE"],
            "size": res.size, "size_unit": res.size_unit,
            "train_time_s": res.train_time_s,
        })
    results_table = pd.DataFrame(rows).set_index("model")
    results_table.to_csv(out / "test_results.csv")
    log(results_table[["RMSE", "MAE", "R2", "train_RMSE", "val_RMSE", "size", "train_time_s"]]
        .round(3).to_string())

    test_df = df.iloc[split.test].reset_index(drop=True)
    pred_df = test_df[["mix_id", *C.MIX_COLUMNS, "age", C.TARGET]].copy()
    for name, p in test_preds.items():
        pred_df[f"pred_{name}"] = p
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    by_age = E.group_breakdown(test_df["age_group"], data.y_test, test_preds)
    by_strength = E.group_breakdown(test_df["strength_group"], data.y_test, test_preds)
    by_age.to_csv(out / "errors_by_age_group.csv", index=False)
    by_strength.to_csv(out / "errors_by_strength_group.csv", index=False)

    # Failure cases: hardest test samples and the effect of extrapolation
    train_df = df.iloc[split.train]
    worst = E.worst_cases(test_df, train_df, data.y_test, test_preds, C.FEATURES)
    worst.to_csv(out / "worst_test_cases.csv", index=False)
    extrap = E.extrapolation_effect(test_df, train_df, data.y_test, test_preds, C.FEATURES)
    extrap.to_csv(out / "errors_inside_vs_outside_train_range.csv", index=False)
    log("\n   Hardest test samples (mean |error| over models):")
    log(worst[["mix_id", "age", "w_c", "actual", "mean_abs_err", "outside_train_range"]]
        .head(8).round(2).to_string(index=False))
    log("\n   RMSE inside vs. outside training input range:")
    log(extrap.pivot(index="model", columns="train_range", values="RMSE").round(2).to_string())

    # 5) Figures ------------------------------------------------------------
    log("\n== 5. Figures")
    P.plot_test_metrics(results_table, figs / "test_metrics.png")
    P.plot_pred_vs_actual(data.y_test, test_preds, figs / "pred_vs_actual_test.png")
    P.plot_residuals(data.y_test, test_preds, figs / "residuals_test.png")
    histories = {n: r.history for n, r in results.items() if r.history}
    if histories:
        P.plot_loss_curves(histories, figs / "loss_curves.png")
    P.plot_group_errors(by_age, C.AGE_LABELS, figs / "errors_by_age_group.png",
                        "Test RMSE by curing-age group", "Curing age")
    P.plot_group_errors(by_strength, C.STRENGTH_LABELS, figs / "errors_by_strength_group.png",
                        "Test RMSE by actual strength range", "Actual strength")

    # 6) Robustness: repeated group splits ----------------------------------
    repeat_summary = None
    if not args.no_repeats:
        seeds = C.REPEAT_SEEDS[:2] if args.fast else C.REPEAT_SEEDS
        log(f"\n== 6. Repeated group splits, seeds {list(seeds)} (best params from seed {C.MAIN_SEED})")
        rep_rows = []
        for seed in seeds:
            data_s = D.make_arrays(df, D.group_split(df, seed))
            for name in model_names:
                res = M.fit(name, best_params[name], seed, data_s, fast=args.fast)
                m = E.regression_metrics(data_s.y_test, res.model.predict(data_s.X_test))
                rep_rows.append({"seed": seed, "model": name, **m})
            log(f"   seed {seed} done")
        rep = pd.DataFrame(rep_rows)
        rep.to_csv(out / "repeated_splits_raw.csv", index=False)
        repeat_summary = rep.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
        repeat_summary.columns = [f"{a}_{b}" for a, b in repeat_summary.columns]
        repeat_summary = repeat_summary.reindex(model_names)
        repeat_summary.to_csv(out / "repeated_splits_summary.csv")
        log(repeat_summary.round(3).to_string())
        P.plot_repeat_summary(repeat_summary, figs / "repeated_splits_rmse.png")

    # 7) Markdown summary ---------------------------------------------------
    write_summary(out, report, summary_split, results_table, best_params, by_age, repeat_summary,
                  worst, extrap, args)
    log(f"\nDone in {time.perf_counter() - t_start:.0f} s. Results in {out}")


def write_summary(out, report, split_df, results_table, best_params, by_age, repeat_summary,
                  worst, extrap, args):
    lines = ["# Experiment summary", ""]
    if args.fast:
        lines += ["> **FAST smoke-test run** - grids and epochs reduced; numbers are not final.", ""]
    lines += ["## Data", "", "```json", json.dumps(report, indent=2), "```", ""]
    lines += ["## Mix-grouped split", "", E.md_table(split_df), ""]
    tbl = results_table.reset_index()[["model", "RMSE", "MAE", "R2", "train_RMSE", "val_RMSE",
                                       "size", "size_unit", "train_time_s"]]
    lines += ["## Test-set results (single evaluation)", "", E.md_table(tbl), ""]
    lines += ["## Best hyperparameters (chosen on validation RMSE)", ""]
    lines += [f"- **{k}**: `{v}`" for k, v in best_params.items()] + [""]
    pivot = by_age.pivot(index="model", columns="group", values="RMSE").reindex(results_table.index)
    lines += ["## Test RMSE by curing-age group (MPa)", "", E.md_table(pivot.reset_index()), ""]
    ex = extrap.pivot(index="model", columns="train_range", values="RMSE").reindex(results_table.index)
    n_out = int(extrap.loc[extrap["train_range"] == "outside", "n"].max()) if (extrap["train_range"] == "outside").any() else 0
    lines += [f"## Extrapolation: test RMSE inside vs. outside training input range ({n_out} rows outside)", "",
              E.md_table(ex.reset_index()), ""]
    cols = ["mix_id", "age", "w_c", "actual", "mean_abs_err", "outside_train_range"]
    lines += ["## Hardest test samples (largest mean |error| over all models)", "",
              E.md_table(worst[cols].head(10)), ""]
    if repeat_summary is not None:
        lines += ["## Repeated group splits (mean / std over seeds)", "",
                  E.md_table(repeat_summary.reset_index()), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
