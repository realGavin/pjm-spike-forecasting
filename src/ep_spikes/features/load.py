"""Load features: DA forecast (as-of issue <= origin) + lagged metered."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_features_for_day(
    metered: pd.Series,
    da_forecast: pd.DataFrame,
    target_hours_utc: pd.DatetimeIndex,
    origin_utc: pd.Timestamp,
    *,
    lags_hours: list[int],
) -> pd.DataFrame:
    """Per-hour features built under the forecast-origin info set.

    Args:
        metered: UTC-indexed hourly zonal metered load (MW).
        da_forecast: cols [target_utc, issue_utc, forecast_mw, zone]; as-of filtered here.
        target_hours_utc: UTC hourly index for day d.
        origin_utc: forecast origin timestamp (10 AM local d-1).
        lags_hours: lags relative to the target hour.
    """
    out = pd.DataFrame(index=target_hours_utc)

    as_of = da_forecast[da_forecast["issue_utc"] <= origin_utc].copy()
    if len(as_of):
        as_of = as_of.sort_values(["target_utc", "issue_utc"])
        as_of = as_of.drop_duplicates("target_utc", keep="last")
        fcst_map = pd.Series(
            as_of["forecast_mw"].to_numpy(),
            index=pd.DatetimeIndex(as_of["target_utc"]),
        )
        out["load_fcst_mw"] = np.array([
            float(fcst_map.get(ts, np.nan)) for ts in target_hours_utc
        ])
    else:
        out["load_fcst_mw"] = np.nan

    pre_origin_metered = metered.loc[metered.index < origin_utc]
    last_observed = float(pre_origin_metered.iloc[-1]) if len(pre_origin_metered) else np.nan
    roll_24 = pre_origin_metered.iloc[-24:] if len(pre_origin_metered) >= 1 else pre_origin_metered
    out["load_last_observed"] = last_observed
    out["load_roll_mean_24h"] = float(roll_24.mean()) if len(roll_24) else np.nan
    out["load_roll_max_24h"] = float(roll_24.max()) if len(roll_24) else np.nan

    for lag in lags_hours:
        lagged_index = target_hours_utc - pd.Timedelta(hours=lag)
        vals = np.array([
            float(pre_origin_metered.get(ts, np.nan)) if ts in pre_origin_metered.index else np.nan
            for ts in lagged_index
        ])
        out[f"load_lag_{lag}h"] = vals

    if "load_fcst_mw" in out.columns and pd.notna(last_observed):
        out["load_fcst_over_last"] = out["load_fcst_mw"] / max(last_observed, 1.0)

    return out
