"""Orchestrator: build the hourly feature panel for a node.

Walk days d in [start, end]; for each day compute forecast_origin, slice the
target window, compute features per block, and stitch the daily frames together.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Settings
from ..tzutil import (
    PJM_TZ,
    forecast_origin_utc,
    local_hours_for_day,
    season_of,
)
from .calendar import calendar_features
from .load import load_features_for_day
from .prices import price_features_for_day
from .storm import storm_features_for_day
from .weather import weather_features_for_day

log = logging.getLogger(__name__)


def build_panel(
    node: str,
    prices: pd.DataFrame,
    metered_load: pd.DataFrame,
    da_load_forecast: pd.DataFrame,
    weather: pd.DataFrame,
    storms: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Assemble the feature panel for one node over the full date range.

    Inputs are UTC-indexed (except storms which have begin_utc column).
    Returns a UTC-indexed DataFrame, one row per hour.
    """
    price_s = prices["lmp"]
    metered_s = metered_load["load_mw"] if "load_mw" in metered_load else pd.Series(dtype=float)

    pjm_start = pd.Timestamp(settings.data.start_date).tz_localize(PJM_TZ)
    pjm_end = pd.Timestamp(settings.data.end_date).tz_localize(PJM_TZ)
    days = pd.date_range(start=pjm_start.normalize(), end=pjm_end.normalize(), freq="D", tz=PJM_TZ)

    feat_cfg = settings.features
    frames: list[pd.DataFrame] = []

    for day_idx, day_local in enumerate(days):
        day_d = day_local.tz_localize(None)
        target = local_hours_for_day(day_d)
        origin = forecast_origin_utc(day_d, origin_hour_local=settings.data.forecast_origin_hour_local)

        cal = calendar_features(target, season_map=settings.labels.season_map)
        px = price_features_for_day(
            price_s, target, origin,
            lags_hours=feat_cfg.price_lags_hours,
            roll_windows_hours=feat_cfg.price_roll_windows_hours,
        )
        lo = load_features_for_day(
            metered_s, da_load_forecast, target, origin,
            lags_hours=feat_cfg.load_lags_hours,
        )
        wx = weather_features_for_day(
            weather, target, origin,
            hourly_vars=feat_cfg.weather_hourly_vars,
            anomaly_col=("temperature_2m_anomaly_z" if feat_cfg.include_weather_anomaly else None),
        )
        st = storm_features_for_day(storms, target, origin, windows_hours=feat_cfg.storm_windows_hours)

        day_frame = pd.concat([cal, px, lo, wx, st], axis=1)
        day_frame["node"] = node
        day_frame["origin_utc"] = origin
        day_frame["lmp"] = price_s.reindex(target).to_numpy()
        day_frame["metered_load"] = metered_s.reindex(target).to_numpy() if len(metered_s) else pd.NA
        frames.append(day_frame)

        if (day_idx + 1) % 30 == 0:
            log.info("build_panel %s: processed %d/%d days", node, day_idx + 1, len(days))

    panel = pd.concat(frames).sort_index()
    panel = panel[~panel.index.duplicated(keep="first")]
    panel.index.name = "timestamp_utc"
    return panel
