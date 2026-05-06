"""Weather features for each target hour of day d.

MVP proxy: Open-Meteo realized observations stand in for the day-d forecast that
would have been issued at 10 AM local d-1. This biases AUCPR upward; flagged in
the report. Phase 2 swaps in Meteomatics issued forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weather_features_for_day(
    weather: pd.DataFrame,
    target_hours_utc: pd.DatetimeIndex,
    origin_utc: pd.Timestamp,      # noqa: ARG001 -- kept for parity with Phase 2 signature
    *,
    hourly_vars: list[str],
    anomaly_col: str | None = "temperature_2m_anomaly_z",
) -> pd.DataFrame:
    """Return one row per target hour, with weather vars + daily aggregates.

    Under MVP (realized-obs proxy), we read `weather` at `target_hours_utc`.
    """
    out = pd.DataFrame(index=target_hours_utc)
    for var in hourly_vars:
        if var not in weather.columns:
            out[var] = np.nan
            continue
        vals = weather[var].reindex(target_hours_utc)
        out[var] = vals.to_numpy()

    if anomaly_col and anomaly_col in weather.columns:
        out[anomaly_col] = weather[anomaly_col].reindex(target_hours_utc).to_numpy()

    if "temperature_2m" in weather.columns:
        day_vals = weather["temperature_2m"].reindex(target_hours_utc).dropna()
        out["temp_daily_max"] = float(day_vals.max()) if len(day_vals) else np.nan
        out["temp_daily_min"] = float(day_vals.min()) if len(day_vals) else np.nan
        out["temp_daily_range"] = out["temp_daily_max"] - out["temp_daily_min"]
    if "precipitation" in weather.columns:
        pr_vals = weather["precipitation"].reindex(target_hours_utc).dropna()
        out["precip_daily_sum"] = float(pr_vals.sum()) if len(pr_vals) else np.nan
    if "wind_speed_10m" in weather.columns:
        ws = weather["wind_speed_10m"].reindex(target_hours_utc).dropna()
        out["wind_daily_max"] = float(ws.max()) if len(ws) else np.nan

    return out
