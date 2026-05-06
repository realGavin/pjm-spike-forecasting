"""Timezone + forecast-origin helpers. PJM settles in America/New_York (EPT, DST-aware)."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

PJM_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def to_utc(ts: pd.DatetimeIndex | pd.Series, src_tz: str = "America/New_York") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx = idx.tz_localize(src_tz, nonexistent="shift_forward", ambiguous="infer")
    return idx.tz_convert("UTC")


def to_pjm(ts: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(PJM_TZ)


def forecast_origin_utc(day_d: pd.Timestamp, origin_hour_local: int = 10) -> pd.Timestamp:
    """10:00 America/New_York on day (d-1), in UTC."""
    d = pd.Timestamp(day_d).tz_localize(None).normalize()
    prev = d - pd.Timedelta(days=1)
    local = pd.Timestamp(
        year=prev.year, month=prev.month, day=prev.day,
        hour=origin_hour_local, tz=PJM_TZ,
    )
    return local.tz_convert("UTC")


def target_window_utc(day_d: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [start, end) for the 24h (or 23/25 on DST) local day d."""
    d = pd.Timestamp(day_d).tz_localize(None).normalize()
    start_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=0, tz=PJM_TZ)
    end_local = (d + pd.Timedelta(days=1))
    end_local = pd.Timestamp(
        year=end_local.year, month=end_local.month, day=end_local.day, hour=0, tz=PJM_TZ,
    )
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def expected_hour_count(day_d: pd.Timestamp) -> int:
    """23 on spring-forward, 25 on fall-back, 24 otherwise."""
    start, end = target_window_utc(day_d)
    return int((end - start) / pd.Timedelta(hours=1))


def local_hours_for_day(day_d: pd.Timestamp) -> pd.DatetimeIndex:
    """UTC hourly index covering the local day d (23/24/25 entries)."""
    start, end = target_window_utc(day_d)
    return pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h", tz="UTC")


def season_of(month: int, season_map: dict[str, list[int]]) -> str:
    for name, months in season_map.items():
        if month in months:
            return name
    raise ValueError(f"Month {month} not in any season in {season_map!r}")
