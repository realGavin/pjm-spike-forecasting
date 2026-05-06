"""Open-Meteo historical archive fetch. Free, no auth.

MVP caveat: returns realized observations, not issued-at-d-1 forecasts.
Used as a proxy; Phase 2 substitutes Meteomatics issued forecasts.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
]


def fetch_openmeteo_year(
    lat: float,
    lon: float,
    year: int,
    *,
    variables: list[str] | None = None,
    timezone: str = "UTC",
    max_retries: int = 4,
) -> pd.DataFrame:
    """One-year hourly pull. Returns UTC-indexed frame, one column per variable."""
    variables = variables or DEFAULT_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(variables),
        "timezone": timezone,
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            log.warning("Open-Meteo %s-%d failed (%s). Retrying in %ds", year, attempt, e, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Open-Meteo fetch failed after {max_retries} retries: {last_err}")

    hourly = data["hourly"]
    df = pd.DataFrame({k: hourly[k] for k in ["time", *variables]})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index.name = "timestamp_utc"
    return df


def fetch_openmeteo_range(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path,
    site_name: str,
    variables: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch a date range, chunked by calendar year, with per-year parquet cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    years = list(range(start.year, end.year + 1))

    frames: list[pd.DataFrame] = []
    for y in years:
        fp = cache_dir / f"openmeteo_{site_name}_{y}.parquet"
        if use_cache and fp.exists():
            log.info("Weather cache hit: %s", fp.name)
            frames.append(pd.read_parquet(fp))
            continue
        log.info("Fetching Open-Meteo %s %d", site_name, y)
        df = fetch_openmeteo_year(lat, lon, y, variables=variables)
        df.to_parquet(fp)
        frames.append(df)

    full = pd.concat(frames).sort_index()
    mask = (full.index >= pd.Timestamp(start_date, tz="UTC")) & (
        full.index < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    )
    return full.loc[mask]


def build_climatology(
    df: pd.DataFrame, column: str = "temperature_2m", tz: str = "America/New_York",
) -> pd.DataFrame:
    """Per (day-of-year, hour-of-day) mean+std from the available history.

    Returns frame indexed by (doy, hod) with columns {mean, std}. Used to compute
    temperature anomaly Z-scores. MVP uses the full available range rather than
    a fixed 30yr baseline; Phase 2 swaps in ERA5 1991-2020 climatology.
    """
    local = df.tz_convert(tz)
    key = pd.DataFrame({
        "doy": local.index.dayofyear,
        "hod": local.index.hour,
        "val": df[column].to_numpy(),
    })
    clim = key.groupby(["doy", "hod"])["val"].agg(["mean", "std"]).fillna(0.0)
    return clim


def attach_temperature_anomaly(
    df: pd.DataFrame, clim: pd.DataFrame, column: str = "temperature_2m",
    tz: str = "America/New_York", out_col: str = "temperature_2m_anomaly_z",
) -> pd.DataFrame:
    local = df.tz_convert(tz)
    key = pd.DataFrame({"doy": local.index.dayofyear, "hod": local.index.hour}, index=df.index)
    joined = key.join(clim, on=["doy", "hod"])
    std = joined["std"].replace(0.0, 1e-6)
    anomaly = (df[column] - joined["mean"]) / std
    out = df.copy()
    out[out_col] = anomaly.to_numpy()
    return out
