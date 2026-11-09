"""All figures. One consistent style; each model keeps the same colour everywhere."""
import math

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from .evaluate import regression_metrics

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = dict(zip(C.MODEL_ORDER, SERIES))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.labelsize": 9,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 2,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _suptitle(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- EDA
def plot_target_distribution(df, path):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(df[C.TARGET], bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("Compressive strength (MPa)")
    ax.set_ylabel("Samples")
    ax.grid(axis="x", visible=False)
    _suptitle(fig, "Distribution of compressive strength")
    _save(fig, path)


def plot_strength_vs_age(df, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    jitter = np.exp(rng.uniform(-0.06, 0.06, len(df)))  # spread overlapping discrete ages
    ax.scatter(df["age"] * jitter, df[C.TARGET], s=18, color=SERIES[0], alpha=0.55,
               edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ages = sorted(df["age"].unique())
    shown = [a for a in ages if a in (1, 3, 7, 14, 28, 56, 90, 180, 365)]
    ax.set_xticks(shown, [str(a) for a in shown])
    ax.minorticks_off()
    ax.set_xlabel("Curing age (days, log scale)")
    ax.set_ylabel("Compressive strength (MPa)")
    _suptitle(fig, "Strength rises with curing age")
    _save(fig, path)


def plot_correlation(df, path):
    cols = C.FEATURES + ["age", C.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    _suptitle(fig, "Pearson correlation of inputs and strength")
    _save(fig, path)


# ---------------------------------------------------------------- model diagnostics
def _grid_axes(n, w=3.4, h=3.3, ncols=3, **kw):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False, **kw)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def plot_pred_vs_actual(y, preds, path, subset="test"):
    names = list(preds)
    fig, axes = _grid_axes(len(names), sharex=True, sharey=True)
    hi = max(y.max(), max(p.max() for p in preds.values())) * 1.05
    for ax, name in zip(axes, names):
        p = preds[name]
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(y, p, s=20, color=MODEL_COLORS[name], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.6, zorder=2)
        m = regression_metrics(y, p)
        ax.set_title(name)
        ax.text(0.04, 0.96, f"RMSE {m['RMSE']:.2f} MPa\nR² {m['R2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
    fig.supxlabel("Actual strength (MPa)", fontsize=9, color=INK2)
    fig.supylabel("Predicted strength (MPa)", fontsize=9, color=INK2)
    _suptitle(fig, f"Predicted vs. actual strength ({subset} set)")
    fig.tight_layout()
    _save(fig, path)


def plot_residuals(y, preds, path, subset="test"):
    names = list(preds)
    fig, axes = _grid_axes(len(names), h=2.9, sharex=True, sharey=True)
    lim = max(np.abs(p - y).max() for p in preds.values()) * 1.1
    for ax, name in zip(axes, names):
        p = preds[name]
        ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(p, y - p, s=20, color=MODEL_COLORS[name], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.6, zorder=2)
        ax.set_ylim(-lim, lim)
        ax.set_title(name)
    fig.supxlabel("Predicted strength (MPa)", fontsize=9, color=INK2)
    fig.supylabel("Residual: actual − predicted (MPa)", fontsize=9, color=INK2)
    _suptitle(fig, f"Residuals ({subset} set)")
    fig.tight_layout()
    _save(fig, path)


def plot_loss_curves(histories, path):
    names = list(histories)
    fig, axes = plt.subplots(1, len(names), figsize=(3.8 * len(names), 3.3), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], names):
        h = histories[name]
        ep = np.arange(1, len(h["rmse"]) + 1)
        ax.plot(ep, h["rmse"], color=SERIES[0], lw=1.5, label="Training")
        ax.plot(ep, h["val_rmse"], color=SERIES[1], lw=1.5, label="Validation")
        ax.axvline(h["best_epoch"], color=MUTED, lw=1, ls=":")
        ax.annotate(f"best epoch {h['best_epoch']}", xy=(h["best_epoch"], 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top", fontsize=7.5, color=INK2)
        ax.set_yscale("log")
        ax.set_yticks([3, 4, 6, 10, 20, 40], ["3", "4", "6", "10", "20", "40"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(name, pad=14)
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("RMSE (MPa, log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.0))
    _suptitle(fig, "Training and validation loss curves")
    fig.tight_layout()
    _save(fig, path)


def plot_test_metrics(table, path):
    """table: DataFrame indexed by model with RMSE, MAE, R2 columns."""
    names = list(table.index)
    ypos = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.45 * len(names) + 1.4), sharey=True)
    for ax, metric, label in zip(axes, ["RMSE", "MAE", "R2"],
                                 ["RMSE (MPa) ↓", "MAE (MPa) ↓", "R² ↑"]):
        vals = table[metric].to_numpy()
        ax.barh(ypos, vals, height=0.62, color=[MODEL_COLORS[n] for n in names])
        fmt = "{:.3f}" if metric == "R2" else "{:.2f}"
        for y_, v in zip(ypos, vals):
            ax.text(v, y_, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(vals.max(), 0) * 1.2 if metric != "R2" else 1.12)
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, names)
    axes[0].invert_yaxis()
    _suptitle(fig, "Test-set performance")
    fig.tight_layout()
    _save(fig, path)


def plot_group_errors(bd, groups, path, title, xlabel):
    models = [m for m in C.MODEL_ORDER if m in set(bd["model"])]
    groups = [g for g in groups if g in set(bd["group"])]
    x = np.arange(len(groups))
    width = 0.84 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, m in enumerate(models):
        sub = bd[bd["model"] == m].set_index("group").reindex(groups)
        ax.bar(x - 0.42 + width * (i + 0.5), sub["RMSE"], width=width,
               color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1, label=m)
    counts = bd.drop_duplicates("group").set_index("group")["n"]
    ax.set_xticks(x, [f"{g}\n(n={counts[g]})" for g in groups])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (MPa)")
    ax.legend(ncol=len(models), loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1, columnspacing=1)
    _suptitle(fig, title)
    fig.tight_layout()
    _save(fig, path)


def plot_repeat_summary(summary, path):
    """summary: DataFrame indexed by model with RMSE_mean, RMSE_std."""
    names = list(summary.index)
    ypos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(names) + 1.3))
    for y_, n in zip(ypos, names):
        mu, sd = summary.loc[n, "RMSE_mean"], summary.loc[n, "RMSE_std"]
        ax.errorbar(mu, y_, xerr=sd, fmt="o", ms=7, color=MODEL_COLORS[n],
                    ecolor=MODEL_COLORS[n], elinewidth=2, capsize=0, mec=SURFACE, mew=1.5)
        ax.text(mu + sd, y_, f"  {mu:.2f} ± {sd:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test RMSE (MPa), mean ± std over seeds")
    ax.set_xlim(0, (summary["RMSE_mean"] + summary["RMSE_std"]).max() * 1.3)
    _suptitle(fig, "Robustness across repeated group splits")
    fig.tight_layout()
    _save(fig, path)
