"""Calendar features. Derived from PJM-local timestamps."""
from __future__ import annotations

import numpy as np
import pandas as pd

import holidays

from ..tzutil import PJM_TZ, season_of

US_HOLIDAYS = holidays.country_holidays("US")


def calendar_features(
    idx_utc: pd.DatetimeIndex,
    season_map: dict[str, list[int]],
) -> pd.DataFrame:
    """Inputs are UTC. Output uses PJM-local wall-clock for hour/dow/month/holiday/season."""
    local = idx_utc.tz_convert(PJM_TZ)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    doy = local.dayofyear.to_numpy()
    is_weekend = (dow >= 5).astype("int8")
    is_holiday = np.array([d.date() in US_HOLIDAYS for d in local]).astype("int8")
    seasons = np.array([season_of(m, season_map) for m in month])
    season_codes = pd.Series(seasons).astype("category").cat.codes.astype("int8").to_numpy()
    sin_h = np.sin(2 * np.pi * hour / 24.0)
    cos_h = np.cos(2 * np.pi * hour / 24.0)
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)

    return pd.DataFrame({
        "hour": hour.astype("int8"),
        "dow": dow.astype("int8"),
        "month": month.astype("int8"),
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "season_code": season_codes,
        "season_name": seasons,
        "hour_sin": sin_h,
        "hour_cos": cos_h,
        "doy_sin": sin_doy,
        "doy_cos": cos_doy,
    }, index=idx_utc)
