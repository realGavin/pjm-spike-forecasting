"""Extract XGBoost feature importances from the last-run model.

Run after the full pipeline completes; writes a markdown table with top-20 features
per fold/node and saves a bar chart figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"


def feature_importance_for_node(node: str, fold_train: tuple[str, str]) -> pd.DataFrame | None:
    """Refit XGBoost on the final fold's training window and extract importances.

    Not a strict reproducibility path (we refit rather than serialize) - this is a
    lightweight, interpretability-focused post-hoc pass.
    """
    panel_fp = PROC / f"panel_{node.lower()}.parquet"
    if not panel_fp.exists():
        return None
    panel = pd.read_parquet(panel_fp)

    from ep_spikes.tzutil import PJM_TZ
    local = panel.index.tz_convert(PJM_TZ)
    start_t = pd.Timestamp(fold_train[0]).tz_localize(PJM_TZ)
    end_t = pd.Timestamp(fold_train[1]).tz_localize(PJM_TZ) + pd.Timedelta(days=1)
    mask = (local >= start_t) & (local < end_t)
    sub = panel.loc[mask].dropna(subset=["lmp"])

    # Build label_either using train-window seasonal threshold
    from ep_spikes.labels.spike import compute_seasonal_thresholds, make_labels
    seasons = sub["season_name"].astype(str)
    thr = compute_seasonal_thresholds(sub["lmp"], seasons, q=0.95)
    labels = make_labels(sub["lmp"], seasons, absolute_threshold=300.0, seasonal_thresholds=thr)
    y = labels["label_either"].to_numpy()

    drop = {"lmp", "metered_load", "node", "origin_utc",
            "label_abs", "label_rel", "label_either", "season_name"}
    feat_cols = [c for c in sub.columns if c not in drop]
    X = sub[feat_cols].select_dtypes(include=[np.number]).fillna(0.0)

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    spw = max(neg / max(pos, 1), 1.0)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=spw, n_jobs=-1, random_state=290, eval_metric="aucpr",
    )
    model.fit(X.to_numpy(), y)

    imp = pd.DataFrame({
        "feature": X.columns,
        "gain_norm": model.feature_importances_,
    }).sort_values("gain_norm", ascending=False).reset_index(drop=True)
    imp["node"] = node
    return imp.head(20)


def make_bar_figure(imp: pd.DataFrame, out: Path, title: str) -> Path:
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["gain_norm"], color="#003262")
    ax.set_xlabel("XGBoost gain importance (normalized)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    # Use the last rolling fold's training window for interpretation
    train_window = ("2021-01-01", "2023-12-31")
    all_imp: list[pd.DataFrame] = []
    for node in ["COMED", "PECO"]:
        imp = feature_importance_for_node(node, train_window)
        if imp is None or imp.empty:
            continue
        all_imp.append(imp)
        make_bar_figure(imp, FIGS / f"feature_importance_{node.lower()}.png",
                        f"Top XGBoost features - {node} (train 2021-2023)")

    if all_imp:
        combined = pd.concat(all_imp, ignore_index=True)
        combined.to_csv(TABLES / "feature_importance_top20.csv", index=False)
        print(f"Wrote {TABLES / 'feature_importance_top20.csv'}")


if __name__ == "__main__":
    main()
