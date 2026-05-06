"""Rolling-origin evaluation protocol.

For each fold:
  1. Slice train and test by calendar date (PJM-local).
  2. Compute seasonal spike thresholds on train rows only.
  3. Apply labels to train+test.
  4. Small time-series CV grid over XGB hyperparams; pick best by AUCPR on val.
  5. Refit on full train.
  6. Predict on test; write preds parquet + metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..config import Fold, Settings
from ..labels.spike import compute_seasonal_thresholds, make_labels
from ..models.logistic import LogisticSpike
from ..models.lear import LEARSpike
from ..models.mlp_spike import LSTMSpike, MLPSpike
from ..models.xgb_quantile import XGBQuantile
from ..models.xgb_spike import XGBSpike
from ..tzutil import PJM_TZ
from .metrics import quantile_metrics, reliability_curve, spike_metrics

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_id: str
    node: str
    model_name: str
    task: str                       # 'spike' | 'quantile'
    label: str                      # 'label_abs' | 'label_rel' | 'label_either' | 'price'
    metrics: dict[str, float]
    best_params: dict[str, Any]
    preds_path: Path
    reliability_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


FEATURE_DROP_COLS = {
    "lmp", "metered_load", "node", "origin_utc",
    "label_abs", "label_rel", "label_either",
    "season_name",   # string label, keep season_code
}


# Column-prefix tags for ablation feature sets.
FEATURE_SET_PREFIXES = {
    "market": ("price_", "load_"),                     # price + load only
    "weather_only": ("temperature_", "apparent_", "relative_", "precipitation",
                     "wind_", "temp_", "precip_"),
    "storm_only": ("storm_",),
    "calendar": ("hour", "dow", "month", "is_weekend", "is_holiday",
                 "season_code", "doy_"),
}


def _feature_columns(panel: pd.DataFrame, *, feature_sets: list[str] | None = None) -> list[str]:
    cols = [c for c in panel.columns if c not in FEATURE_DROP_COLS]
    if feature_sets is None:
        return cols
    # Calendar features are always included; everything else is gated by the requested sets
    cal_prefixes = FEATURE_SET_PREFIXES["calendar"]
    allowed_prefixes = list(cal_prefixes)
    for s in feature_sets:
        if s == "all":
            return cols
        allowed_prefixes.extend(FEATURE_SET_PREFIXES.get(s, ()))
    def _keep(c: str) -> bool:
        return any(c.startswith(p) for p in allowed_prefixes)
    return [c for c in cols if _keep(c)]


def _date_mask(panel: pd.DataFrame, start: str, end: str) -> np.ndarray:
    local = panel.index.tz_convert(PJM_TZ)
    d_start = pd.Timestamp(start).tz_localize(PJM_TZ)
    d_end = pd.Timestamp(end).tz_localize(PJM_TZ) + pd.Timedelta(days=1)
    mask = (local >= d_start) & (local < d_end)
    return np.asarray(mask)


def _test_mask_for_fold(panel: pd.DataFrame, fold: Fold) -> np.ndarray:
    year_spec = fold.test_year
    if "--" in year_spec:
        # smoke format like "2023-08--09" = Aug 1 through Sep 30 2023
        parts = year_spec.split("-")
        year, m_lo, m_hi = int(parts[0]), int(parts[1]), int(parts[3])
        start = f"{year}-{m_lo:02d}-01"
        last_day = pd.Timestamp(year=year, month=m_hi, day=1) + pd.offsets.MonthEnd(0)
        end = last_day.strftime("%Y-%m-%d")
    else:
        year = int(year_spec)
        start = f"{year}-01-01"
        end = f"{year}-12-31"
    return _date_mask(panel, start, end)


def _season_series(panel: pd.DataFrame) -> pd.Series:
    if "season_name" in panel.columns:
        return panel["season_name"].astype(str)
    raise KeyError("panel missing 'season_name' column from calendar features")


def run_spike_fold(
    panel: pd.DataFrame,
    fold: Fold,
    node: str,
    settings: Settings,
    *,
    label_col: str = "label_either",
    out_preds_dir: Path,
    out_fig_dir: Path,
    extra_models: list[str] | None = None,
    feature_sets: list[str] | None = None,
    tag_suffix: str = "",
) -> list[FoldResult]:
    """Train + evaluate spike models for one fold.

    extra_models may include: 'lear', 'mlp', 'lstm'. Default trains logistic + xgb_spike.
    feature_sets masks features via prefix (e.g. ['market'] for market-only ablation).
    tag_suffix is appended to fold_id in artifact names (useful for ablation runs).
    """
    fold_id = fold.fold_id + (("_" + tag_suffix) if tag_suffix else "")
    log.info("Fold %s: starting spike task", fold_id)
    seasons = _season_series(panel)
    train_mask = _date_mask(panel, fold.train_start, fold.train_end)
    test_mask = _test_mask_for_fold(panel, fold)

    thresholds = compute_seasonal_thresholds(
        panel.loc[train_mask, "lmp"], seasons.loc[train_mask], q=settings.labels.seasonal_quantile,
    )
    labels = make_labels(
        panel["lmp"], seasons,
        absolute_threshold=settings.labels.absolute_threshold,
        seasonal_thresholds=thresholds,
    )
    for c in labels.columns:
        panel[c] = labels[c].to_numpy()

    feat_cols = _feature_columns(panel, feature_sets=feature_sets)
    X_train = panel.loc[train_mask, feat_cols].select_dtypes(include=[np.number])
    y_train = pd.Series(panel.loc[train_mask, label_col].to_numpy(), index=X_train.index)
    X_test = panel.loc[test_mask, feat_cols].reindex(columns=X_train.columns)
    y_test = pd.Series(panel.loc[test_mask, label_col].to_numpy(), index=X_test.index)

    results: list[FoldResult] = []

    # Logistic baseline (no grid search)
    lr_model = LogisticSpike.from_config(settings.models.logistic).fit(X_train.fillna(X_train.median()), y_train)
    p_lr = lr_model.predict_proba(X_test.fillna(X_train.median()))
    results.append(_finalize_spike(
        p_lr, y_test, X_test, "logistic", fold_id, node, label_col,
        best={"C": settings.models.logistic.get("C", 1.0)},
        preds_dir=out_preds_dir, fig_dir=out_fig_dir,
    ))

    # XGBoost with small grid + time-series CV
    spec = XGBSpike.from_config(settings.models.xgb_spike)
    best_params, best_score = None, -np.inf
    if len(spec.param_grid()) == 1:
        best_params = spec.param_grid()[0]
        log.info("XGB grid size 1; skipping CV.")
    else:
        tscv = TimeSeriesSplit(n_splits=settings.models.cv_n_splits)
        for combo in spec.param_grid():
            cv_scores: list[float] = []
            for tr_idx, va_idx in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                m = XGBSpike.from_config(settings.models.xgb_spike).fit(X_tr, y_tr, overrides=combo)
                from sklearn.metrics import average_precision_score
                p_va = m.predict_proba(X_va)
                if len(np.unique(y_va.to_numpy())) < 2:
                    continue
                cv_scores.append(average_precision_score(y_va.to_numpy(), p_va))
            if not cv_scores:
                continue
            mean_score = float(np.mean(cv_scores))
            log.info("XGB combo %s -> CV AUCPR %.4f", combo, mean_score)
            if mean_score > best_score:
                best_score = mean_score
                best_params = combo
        if best_params is None:
            best_params = spec.param_grid()[0]

    xgb_model = XGBSpike.from_config(settings.models.xgb_spike).fit(X_train, y_train, overrides=best_params)
    p_xgb = xgb_model.predict_proba(X_test)
    results.append(_finalize_spike(
        p_xgb, y_test, X_test, "xgb_spike", fold_id, node, label_col,
        best=best_params, preds_dir=out_preds_dir, fig_dir=out_fig_dir,
    ))

    # Phase 2 models — opt in via `extra_models`
    X_train_i = X_train.fillna(X_train.median())
    X_test_i = X_test.fillna(X_train.median())

    if extra_models and "lear" in extra_models:
        try:
            lear_cfg = dict(getattr(settings.models, "lear", {}) or {})
            m = LEARSpike.from_config(lear_cfg).fit(X_train_i, y_train)
            p = m.predict_proba(X_test_i)
            results.append(_finalize_spike(
                p, y_test, X_test, "lear", fold_id, node, label_col,
                best=lear_cfg, preds_dir=out_preds_dir, fig_dir=out_fig_dir,
            ))
        except Exception as e:
            log.warning("LEAR training failed (%s); skipping", e)

    if extra_models and "mlp" in extra_models:
        try:
            mlp_cfg = dict(getattr(settings.models, "mlp", {}) or {})
            m = MLPSpike.from_config(mlp_cfg).fit(X_train_i, y_train)
            p = m.predict_proba(X_test_i)
            results.append(_finalize_spike(
                p, y_test, X_test, "mlp", fold_id, node, label_col,
                best=mlp_cfg, preds_dir=out_preds_dir, fig_dir=out_fig_dir,
            ))
        except Exception as e:
            log.warning("MLP training failed (%s); skipping", e)

    if extra_models and "lstm" in extra_models:
        try:
            lstm_cfg = dict(getattr(settings.models, "lstm", {}) or {})
            m = LSTMSpike.from_config(lstm_cfg).fit(X_train_i, y_train)
            p = m.predict_proba(X_test_i)
            results.append(_finalize_spike(
                p, y_test, X_test, "lstm", fold_id, node, label_col,
                best=lstm_cfg, preds_dir=out_preds_dir, fig_dir=out_fig_dir,
            ))
        except Exception as e:
            log.warning("LSTM training failed (%s); skipping", e)

    return results


def run_quantile_fold(
    panel: pd.DataFrame,
    fold: Fold,
    node: str,
    settings: Settings,
    *,
    out_preds_dir: Path,
) -> FoldResult:
    fold_id = fold.fold_id
    log.info("Fold %s: starting quantile task", fold_id)
    train_mask = _date_mask(panel, fold.train_start, fold.train_end)
    test_mask = _test_mask_for_fold(panel, fold)
    feat_cols = _feature_columns(panel)
    X_train = panel.loc[train_mask, feat_cols].select_dtypes(include=[np.number])
    y_train = pd.Series(panel.loc[train_mask, "lmp"].to_numpy(), index=X_train.index)
    X_test = panel.loc[test_mask, feat_cols].reindex(columns=X_train.columns)
    y_test = pd.Series(panel.loc[test_mask, "lmp"].to_numpy(), index=X_test.index)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    y_train = y_train.dropna()
    X_train = X_train.loc[y_train.index]

    model = XGBQuantile.from_config(settings.models.xgb_quantile).fit(X_train, y_train)
    preds = model.predict_quantiles(X_test)
    preds["lmp"] = y_test.to_numpy()

    alphas = list(settings.models.xgb_quantile.get("quantile_alpha", [0.95, 0.99]))
    metrics = quantile_metrics(y_test.dropna().to_numpy(),
                               preds.dropna(subset=["lmp"]).drop(columns=["lmp"]),
                               alphas=alphas)

    preds_path = out_preds_dir / f"{node}_{fold_id}_quantile.parquet"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(preds_path)

    return FoldResult(
        fold_id=fold_id, node=node, model_name="xgb_quantile",
        task="quantile", label="price",
        metrics=metrics, best_params=settings.models.xgb_quantile,
        preds_path=preds_path,
    )


def _finalize_spike(
    y_prob: np.ndarray,
    y_test: pd.Series,
    X_test: pd.DataFrame,
    model_name: str,
    fold_id: str,
    node: str,
    label_col: str,
    best: dict[str, Any],
    preds_dir: Path,
    fig_dir: Path,
) -> FoldResult:
    metrics = spike_metrics(y_test.to_numpy(), y_prob)
    rel = reliability_curve(y_test.to_numpy(), y_prob, n_bins=10)
    preds_df = pd.DataFrame({"y_true": y_test.to_numpy(), "y_prob": y_prob}, index=X_test.index)
    preds_df["timestamp_utc"] = X_test.index
    preds_path = preds_dir / f"{node}_{fold_id}_{model_name}_{label_col}.parquet"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_parquet(preds_path)
    fig_path = fig_dir / f"{node}_{fold_id}_{model_name}_{label_col}_reliability.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    rel.to_csv(fig_path, index=False)
    return FoldResult(
        fold_id=fold_id, node=node, model_name=model_name, task="spike", label=label_col,
        metrics=metrics, best_params=best, preds_path=preds_path, reliability_path=fig_path,
    )
