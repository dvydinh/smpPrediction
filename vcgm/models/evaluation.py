from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf
from vcgm import config as cfg

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred):
    flat_true, flat_pred = y_true.flatten(), y_pred.flatten()
    result = {
        "overall_rmse": float(np.sqrt(mean_squared_error(flat_true, flat_pred))),
        "overall_mae": float(mean_absolute_error(flat_true, flat_pred)),
    }
    if y_true.ndim == 2:
        result["cycle_rmse"] = [float(np.sqrt(mean_squared_error(y_true[:, k], y_pred[:, k]))) for k in range(y_true.shape[1])]
        result["cycle_mae"] = [float(mean_absolute_error(y_true[:, k], y_pred[:, k])) for k in range(y_true.shape[1])]
    return result


def walk_forward_cv(forecaster_cls, X_daily, Y_daily, dates, feature_names, splits=cfg.CV_SPLITS):
    dates_pd = pd.to_datetime(dates)
    results = []
    for split in splits:
        train_mask = dates_pd <= split["train_end"]
        val_mask = (dates_pd >= split["val_start"]) & (dates_pd <= split["val_end"])
        if val_mask.sum() == 0:
            continue
        logger.info("%s  train=%d  val=%d", split["name"], train_mask.sum(), val_mask.sum())
        fc = forecaster_cls()
        fc.fit(X_daily[train_mask], Y_daily[train_mask], feature_names)
        preds = fc.predict(X_daily[val_mask])
        metrics = compute_metrics(Y_daily[val_mask], preds)
        metrics["fold"] = split["name"]
        results.append(metrics)
        logger.info("  RMSE=%.2f  MAE=%.2f", metrics["overall_rmse"], metrics["overall_mae"])
        del fc
    return results


def plot_residuals(y_true, y_pred, save_path=cfg.FIGURE_DIR / "residual_analysis.png"):
    residuals = y_true.flatten() - y_pred.flatten()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    axes[0, 0].scatter(range(len(residuals)), residuals, s=1, alpha=0.3, c="steelblue")
    axes[0, 0].axhline(0, c="red", lw=1); axes[0, 0].set_title("Residuals")

    axes[0, 1].hist(residuals, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(0, c="red", lw=1); axes[0, 1].set_title("Distribution")

    r_acf = acf(residuals, nlags=100)
    ci = 1.96 / np.sqrt(len(residuals))
    axes[1, 0].bar(range(len(r_acf)), r_acf, color="darkorange", alpha=0.7)
    axes[1, 0].axhline(ci, ls="--", c="blue", alpha=0.5)
    axes[1, 0].axhline(-ci, ls="--", c="blue", alpha=0.5)
    axes[1, 0].set_title("ACF")

    if y_true.ndim == 2:
        cycle_rmse = [np.sqrt(mean_squared_error(y_true[:, k], y_pred[:, k])) for k in range(y_true.shape[1])]
        axes[1, 1].bar(range(48), cycle_rmse, color="steelblue", alpha=0.7)
        axes[1, 1].set_xticks(range(0, 48, 4))
        axes[1, 1].set_xticklabels([f"{c // 2:02d}:{(c % 2) * 30:02d}" for c in range(0, 48, 4)], rotation=45)
        axes[1, 1].set_title("RMSE by cycle")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def feature_importance(models, feature_names, top_n=20, save_path=cfg.FIGURE_DIR / "feature_importance.png"):
    gain = np.zeros(len(feature_names))
    for k in range(cfg.CYCLES_PER_DAY):
        gain += models[k].feature_importance(importance_type="gain")
    gain /= cfg.CYCLES_PER_DAY

    ranking = pd.DataFrame({"feature": feature_names, "avg_gain": gain})
    ranking = ranking.sort_values("avg_gain", ascending=False).reset_index(drop=True)

    top = ranking.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top["avg_gain"].values[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax.set_title(f"Top {top_n} features (avg gain)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return ranking


def plot_predictions(y_true, y_pred, dates, n_samples=6, save_path=cfg.FIGURE_DIR / "actual_vs_predicted.png"):
    idx = np.linspace(0, len(dates) - 1, min(n_samples, len(dates)), dtype=int)
    fig, axes = plt.subplots(len(idx), 1, figsize=(16, 4 * len(idx)))
    if len(idx) == 1:
        axes = [axes]

    tick_pos = range(0, 48, 4)
    tick_labels = [f"{c // 2:02d}:{(c % 2) * 30:02d}" for c in tick_pos]

    for ax, i in zip(axes, idx):
        ax.plot(range(48), y_true[i], "b-o", ms=3, lw=1.5, label="Actual", alpha=0.8)
        ax.plot(range(48), y_pred[i], "r--s", ms=3, lw=1.5, label="Predicted", alpha=0.8)
        ax.set_title(pd.Timestamp(dates[i]).strftime("%Y-%m-%d (%A)"), fontweight="bold")
        ax.set(ylabel="SMP (VND)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics(metrics, path=cfg.METRICS_DIR / "metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
