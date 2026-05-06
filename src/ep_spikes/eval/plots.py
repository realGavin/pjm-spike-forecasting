"""Plotting helpers. Matplotlib only."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def reliability_diagram(curve: pd.DataFrame, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
    valid = curve.dropna(subset=["mean_pred", "empirical_rate"])
    if not valid.empty:
        ax.plot(valid["mean_pred"], valid["empirical_rate"], "o-", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def pr_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, out: Path) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label="Model")
    ax.axhline(float(np.mean(y_true)), color="k", linestyle="--", alpha=0.6, label="Prevalence")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def cvar_curve(
    thresholds: list[float],
    cvars: list[float],
    baseline: float,
    title: str,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, cvars, "o-", label="Hedged CVaR95")
    ax.axhline(baseline, color="k", linestyle="--", label="No-hedge baseline")
    ax.set_xlabel("Spike probability threshold")
    ax.set_ylabel("Daily CVaR95 ($)")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
