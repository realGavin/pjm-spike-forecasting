"""Event study: regress DA LMP on weather-event indicators with calendar fixed effects.

Follows the revised-proposal stipulation that this is EDA only, run on the
first training fold, never used for feature selection on later folds.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)

EVENT_FEATURES = [
    "storm_n_events_24h",
    "storm_n_severe_24h",
    "storm_tornado_recent",
    "storm_exc_heat_recent",
    "storm_winter_recent",
    "storm_ice_recent",
    "storm_high_wind_recent",
]


def run_event_study(
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
    *,
    target_col: str = "lmp",
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """OLS with hour-of-day, day-of-week, month dummies + storm indicators."""
    from ..tzutil import PJM_TZ
    local = panel.index.tz_convert(PJM_TZ)
    d_start = pd.Timestamp(train_start).tz_localize(PJM_TZ)
    d_end = pd.Timestamp(train_end).tz_localize(PJM_TZ) + pd.Timedelta(days=1)
    mask = (local >= d_start) & (local < d_end)
    df = panel.loc[mask].copy()
    df = df.dropna(subset=[target_col])
    if df.empty:
        log.warning("Event study: no training rows")
        return pd.DataFrame()

    y = np.log1p(df[target_col].clip(lower=0))  # log-spike-severity approximation
    hour_d = pd.get_dummies(df["hour"], prefix="h", drop_first=True).astype(float)
    dow_d = pd.get_dummies(df["dow"], prefix="d", drop_first=True).astype(float)
    mon_d = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(float)

    event_cols = [c for c in EVENT_FEATURES if c in df.columns]
    events = df[event_cols].astype(float).fillna(0.0)
    temp_cols = [c for c in ("temperature_2m_anomaly_z", "temperature_2m") if c in df.columns]
    temps = df[temp_cols].astype(float).fillna(df[temp_cols].median())

    X = pd.concat([hour_d, dow_d, mon_d, events, temps], axis=1)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HC1")
    rows = []
    for name, coef, se, t, p in zip(
        model.params.index, model.params.to_numpy(),
        model.bse.to_numpy(), model.tvalues.to_numpy(), model.pvalues.to_numpy(),
    ):
        rows.append({
            "variable": name, "coef": float(coef), "std_err": float(se),
            "t": float(t), "p": float(p),
        })
    result = pd.DataFrame(rows)
    result["is_event"] = result["variable"].isin(EVENT_FEATURES + temp_cols)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
        log.info("Event-study coefficients written to %s", out_csv)
        summary_text = str(model.summary())
        summary_fp = out_csv.with_suffix(".txt")
        summary_fp.write_text(summary_text)
    return result
