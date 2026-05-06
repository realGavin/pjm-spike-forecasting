"""Evaluation metrics for spike and quantile tasks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def aucpr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def aucroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins in [0,1]. Returns per-bin empirical rate + mean predicted."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        sel = bin_idx == b
        if not sel.any():
            rows.append({
                "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                "mean_pred": np.nan, "empirical_rate": np.nan, "count": 0,
            })
            continue
        rows.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "mean_pred": float(y_prob[sel].mean()),
            "empirical_rate": float(y_true[sel].mean()),
            "count": int(sel.sum()),
        })
    return pd.DataFrame(rows)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def quantile_coverage(y_true: np.ndarray, y_pred_q: np.ndarray) -> float:
    return float(np.mean(y_true <= y_pred_q))


def spike_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y_true))
    return {
        "prevalence": prevalence,
        "aucpr": aucpr(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "aucroc": aucroc(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": brier(y_true, y_prob),
        "lift_over_prevalence": (aucpr(y_true, y_prob) / prevalence) if prevalence > 0 else float("nan"),
    }


def quantile_metrics(
    y_true: np.ndarray, preds: pd.DataFrame, alphas: list[float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for a in alphas:
        col = f"q{int(a * 100)}"
        if col not in preds.columns:
            continue
        yp = preds[col].to_numpy()
        metrics[f"pinball_{col}"] = pinball_loss(y_true, yp, a)
        metrics[f"coverage_{col}"] = quantile_coverage(y_true, yp)
    return metrics
