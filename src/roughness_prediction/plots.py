from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "figure.facecolor": "#f7f5ef",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#263238",
    "axes.labelcolor": "#263238",
    "xtick.color": "#263238",
    "ytick.color": "#263238",
    "text.color": "#263238",
    "font.family": "DejaVu Sans",
}


def create_result_plots(
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        _plot_predictions(predictions, metrics, output_dir / "prediction_parity.png")
        _plot_importance(importance, output_dir / "feature_importance.png")


def _plot_predictions(predictions: pd.DataFrame, metrics: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    ax.scatter(
        predictions["actual_ra_um"],
        predictions["predicted_ra_um"],
        s=58,
        color="#0f766e",
        alpha=0.86,
        edgecolor="#083f3a",
        linewidth=0.55,
    )
    low = min(predictions["actual_ra_um"].min(), predictions["predicted_ra_um"].min())
    high = max(predictions["actual_ra_um"].max(), predictions["predicted_ra_um"].max())
    pad = (high - low) * 0.08
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#d97706", linewidth=2.2)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title("Surface Roughness Prediction", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Measured Ra (um)")
    ax.set_ylabel("Predicted Ra (um)")
    ax.grid(True, color="#d8d3c4", linewidth=0.8, alpha=0.65)
    ax.text(
        0.03,
        0.95,
        f"MAE: {metrics['mae_um']:.3f} um\nR2: {metrics['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eef7f4", "edgecolor": "#9cc9be"},
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(10).iloc[::-1]
    colors = ["#0f766e" if idx % 2 else "#2563eb" for idx in range(len(top))]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    ax.barh(top["feature"], top["importance_mean"], color=colors, alpha=0.9)
    ax.set_title("Top Signal Features", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Permutation importance")
    ax.grid(axis="x", color="#d8d3c4", linewidth=0.8, alpha=0.65)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
