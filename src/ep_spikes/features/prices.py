"""Price-derived features. Strictly respect `origin` — no post-origin values used."""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_features_for_day(
    price: pd.Series,
    target_hours_utc: pd.DatetimeIndex,
    origin_utc: pd.Timestamp,
    *,
    lags_hours: list[int],
    roll_windows_hours: list[int],
) -> pd.DataFrame:
    """Compute price-derived features for the hours of day d.

    For target hour h on day d:
      - lag_{L}: price at h - L hours (must be < origin)
      - roll_mean_{W}_origin: rolling mean over [origin-W, origin)
      - roll_std_{W}_origin: rolling std over [origin-W, origin)
      - roll_max_{W}_origin: rolling max over [origin-W, origin)
      - last_hour_price: price at the last settled hour (origin - 1h)
      - price_ramp_24h: (price at origin-1h) - (price at origin-25h)
    """
    pre_origin = price.loc[price.index < origin_utc]
    rolling_scalars: dict[str, float] = {}
    for w in roll_windows_hours:
        window = pre_origin.loc[pre_origin.index >= origin_utc - pd.Timedelta(hours=w)]
        rolling_scalars[f"price_roll_mean_{w}h"] = float(window.mean()) if len(window) else np.nan
        rolling_scalars[f"price_roll_std_{w}h"] = float(window.std()) if len(window) > 1 else 0.0
        rolling_scalars[f"price_roll_max_{w}h"] = float(window.max()) if len(window) else np.nan
        rolling_scalars[f"price_roll_min_{w}h"] = float(window.min()) if len(window) else np.nan

    last_hour_ts = origin_utc - pd.Timedelta(hours=1)
    ts_minus_25 = origin_utc - pd.Timedelta(hours=25)
    last_price = float(pre_origin.get(last_hour_ts, np.nan)) if last_hour_ts in pre_origin.index else np.nan
    ref_minus_25 = float(pre_origin.get(ts_minus_25, np.nan)) if ts_minus_25 in pre_origin.index else np.nan
    price_ramp_24h = (last_price - ref_minus_25) if pd.notna(last_price) and pd.notna(ref_minus_25) else np.nan

    out = pd.DataFrame(index=target_hours_utc)
    for col, val in rolling_scalars.items():
        out[col] = val
    out["price_last_hour"] = last_price
    out["price_ramp_24h"] = price_ramp_24h

    for lag in lags_hours:
        lagged_index = target_hours_utc - pd.Timedelta(hours=lag)
        vals = np.array([
            float(pre_origin.get(ts, np.nan)) if ts in pre_origin.index else np.nan
            for ts in lagged_index
        ])
        out[f"price_lag_{lag}h"] = vals
        if lag < 24:
            if (lagged_index >= origin_utc).any():
                raise AssertionError(f"Price lag {lag}h reaches >= origin {origin_utc}")
    return out
