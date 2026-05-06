"""PJM Data Miner 2 fetch - gridstatus-first with direct-REST fallback.

DataMiner2 archives anything older than ~2 years. For archived dates only a
date range filter is permitted; zone/pnode filters are rejected. Strategy:
  - DA LMPs:   gridstatus.PJM.get_lmp (handles archive + filter internally),
               then filter client-side to the requested zone.
  - DA load forecast: gridstatus.PJM.get_load_forecast_historical, returning
               a wide frame with one column per zone.
  - Metered load: direct REST with date-only filter, client-side zone filter.

Requires PJM_API_KEY env var.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

PJM_REST_BASE = "https://api.pjm.com/api/v1"
ROW_COUNT = 50000


def _get_api_key() -> str:
    key = os.environ.get("PJM_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PJM_API_KEY is not set. Add it to .env or export the env var.\n"
            "Get one at https://apiportal.pjm.com/ -> Products -> Data Miner API -> Subscribe."
        )
    return key


def _pjm_client():
    import gridstatus
    return gridstatus.PJM(api_key=_get_api_key())


# ---------- Day-Ahead LMP (via gridstatus) ----------

def fetch_da_lmp_zonal(
    zone: str,
    start: str,
    end: str,
    *,
    cache_dir: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Day-ahead hourly zonal LMP. Per-year parquet cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_t = pd.Timestamp(start)
    end_t = pd.Timestamp(end)

    frames: list[pd.DataFrame] = []
    for year in range(start_t.year, end_t.year + 1):
        fp = cache_dir / f"da_lmp_{zone.lower()}_{year}.parquet"
        if use_cache and fp.exists():
            log.info("PJM DA LMP cache hit: %s", fp.name)
            frames.append(pd.read_parquet(fp))
            continue
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        df = _fetch_da_lmp_via_gridstatus(zone, y_start, y_end)
        if df.empty:
            log.warning("DA LMP empty for %s %d", zone, year)
            continue
        df.to_parquet(fp)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    mask = (full.index >= pd.Timestamp(start, tz="UTC")) & (
        full.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    )
    return full.loc[mask]


def _fetch_da_lmp_via_gridstatus(zone: str, start: str, end: str) -> pd.DataFrame:
    """Pull DA LMP month-by-month; 23 zones × ~720 h ≈ 17K rows per call, avoids pagination."""
    iso = _pjm_client()
    months = pd.date_range(start, (pd.Timestamp(end) + pd.Timedelta(days=1)), freq="MS")
    if len(months) == 0 or months[0] > pd.Timestamp(start):
        months = pd.DatetimeIndex([pd.Timestamp(start)]).append(months)
    months = months.append(pd.DatetimeIndex([pd.Timestamp(end) + pd.Timedelta(days=1)]))
    months = pd.DatetimeIndex(sorted(set(months)))

    frames: list[pd.DataFrame] = []
    for i in range(len(months) - 1):
        s = months[i].strftime("%Y-%m-%d")
        e = months[i + 1].strftime("%Y-%m-%d")
        if s == e:
            continue
        log.info("  DA LMP chunk %s to %s", s, e)
        raw = iso.get_lmp(
            start=s, end=e, market="DAY_AHEAD_HOURLY", location_type="ZONE",
        )
        if raw is None or len(raw) == 0:
            continue
        sub = raw[raw["Location Name"] == zone].copy()
        if sub.empty:
            continue
        utc = pd.to_datetime(sub["Interval Start"], utc=True)
        frames.append(pd.DataFrame({
            "lmp": pd.to_numeric(sub["LMP"], errors="coerce").to_numpy(),
            "congestion_lmp": pd.to_numeric(sub["Congestion"], errors="coerce").to_numpy(),
            "loss_lmp": pd.to_numeric(sub["Loss"], errors="coerce").to_numpy(),
            "zone": zone,
        }, index=pd.DatetimeIndex(utc)))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "timestamp_utc"
    return out


# ---------- Metered Load (direct REST) ----------

def fetch_metered_load_zonal(
    zone: str,
    start: str,
    end: str,
    *,
    cache_dir: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_t = pd.Timestamp(start)
    end_t = pd.Timestamp(end)

    frames: list[pd.DataFrame] = []
    for year in range(start_t.year, end_t.year + 1):
        fp = cache_dir / f"metered_load_{zone.lower()}_{year}.parquet"
        if use_cache and fp.exists():
            log.info("PJM metered load cache hit: %s", fp.name)
            frames.append(pd.read_parquet(fp))
            continue
        df = _fetch_metered_load_rest(zone, f"{year}-01-01", f"{year}-12-31")
        if df.empty:
            log.warning("Metered load empty for %s %d", zone, year)
            continue
        df.to_parquet(fp)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    mask = (full.index >= pd.Timestamp(start, tz="UTC")) & (
        full.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    )
    return full.loc[mask]


METERED_LOAD_ALIASES: dict[str, list[str]] = {
    "COMED": ["CE", "COMED"],
    "PECO":  ["PE", "PECO"],
    "BGE":   ["BC", "BGE"],
    "DPL":   ["DPLCO", "DPL"],
    "PSEG":  ["PS", "PSEG"],
    "AEP":   ["AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT", "AEP"],
}


def _fetch_metered_load_rest(zone: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "rowCount": ROW_COUNT,
        "startRow": 1,
        "datetime_beginning_ept": f"{start} 00:00 to {end} 23:59",
    }
    rows = _paged_get("hrl_load_metered", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    aliases = [a.upper() for a in METERED_LOAD_ALIASES.get(zone.upper(), [zone.upper()])]
    if "load_area" in df.columns:
        df = df[df["load_area"].astype(str).str.upper().isin(aliases)]
    if df.empty:
        return pd.DataFrame()
    sub_areas = df["load_area"].unique().tolist() if "load_area" in df.columns else []
    if len(sub_areas) > 1:
        log.info("Metered load for %s = sum of sub-areas %s", zone, sorted(sub_areas))
        df = df.groupby("datetime_beginning_utc", as_index=False)["mw"].sum()
    utc = pd.to_datetime(df["datetime_beginning_utc"], utc=True)
    out = pd.DataFrame({
        "load_mw": pd.to_numeric(df["mw"], errors="coerce").to_numpy(),
        "zone": zone,
    }, index=pd.DatetimeIndex(utc))
    out.index.name = "timestamp_utc"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# ---------- DA Load Forecast (via gridstatus) ----------

def fetch_da_load_forecast(
    zone: str,
    start: str,
    end: str,
    *,
    cache_dir: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Historical DA load forecast issued at multiple vintages per target hour.

    Returns long-form [target_utc, issue_utc, forecast_mw, zone].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_t = pd.Timestamp(start)
    end_t = pd.Timestamp(end)

    frames: list[pd.DataFrame] = []
    for year in range(start_t.year, end_t.year + 1):
        fp = cache_dir / f"da_load_forecast_{zone.lower()}_{year}.parquet"
        if use_cache and fp.exists():
            log.info("PJM load forecast cache hit: %s", fp.name)
            frames.append(pd.read_parquet(fp))
            continue
        df = _fetch_load_forecast_via_gridstatus(zone, f"{year}-01-01", f"{year}-12-31")
        if df.empty:
            log.warning("DA load forecast empty for %s %d", zone, year)
            continue
        df.to_parquet(fp)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["issue_utc", "target_utc"]).reset_index(drop=True)


def _fetch_load_forecast_via_gridstatus(zone: str, start: str, end: str) -> pd.DataFrame:
    iso = _pjm_client()
    months = pd.date_range(start, (pd.Timestamp(end) + pd.Timedelta(days=1)), freq="MS")
    if len(months) == 0 or months[0] > pd.Timestamp(start):
        months = pd.DatetimeIndex([pd.Timestamp(start)]).append(months)
    months = months.append(pd.DatetimeIndex([pd.Timestamp(end) + pd.Timedelta(days=1)]))
    months = pd.DatetimeIndex(sorted(set(months)))

    frames: list[pd.DataFrame] = []
    for i in range(len(months) - 1):
        s = months[i].strftime("%Y-%m-%d")
        e = months[i + 1].strftime("%Y-%m-%d")
        if s == e:
            continue
        log.info("  load forecast chunk %s to %s", s, e)
        raw = iso.get_load_forecast_historical(start=s, end=e)
        if raw is None or len(raw) == 0:
            continue
        zone_col = zone if zone in raw.columns else (
            zone.upper() if zone.upper() in raw.columns else None
        )
        # PJM DA forecast feed reports MIDATL aggregate rather than PECO/BGE/PPL zones
        if zone_col is None and zone.upper() in {"PECO", "BGE", "PPL", "PEPCO", "DPL", "JCPL", "AECO", "PSEG"}:
            zone_col = "MIDATL"
            log.warning("Zone %s not in forecast columns; using MIDATL regional aggregate "
                        "(limits PECO forecast fidelity).", zone)
        if zone_col is None:
            raise RuntimeError(f"Zone {zone} not in forecast columns: {list(raw.columns)}")
        frames.append(pd.DataFrame({
            "target_utc": pd.to_datetime(raw["Interval Start"], utc=True).to_numpy(),
            "issue_utc": pd.to_datetime(raw["Publish Time"], utc=True).to_numpy(),
            "forecast_mw": pd.to_numeric(raw[zone_col], errors="coerce").to_numpy(),
            "zone": zone,
        }))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).dropna(subset=["target_utc", "issue_utc", "forecast_mw"])


# ---------- Generic paginated GET ----------

def _paged_get(feed: str, params: dict) -> list[dict]:
    key = _get_api_key()
    headers = {"Ocp-Apim-Subscription-Key": key, "accept": "application/json"}
    url = f"{PJM_REST_BASE}/{feed}"

    all_rows: list[dict] = []
    start_row = 1
    while True:
        p = dict(params)
        p["startRow"] = start_row
        for attempt in range(5):
            try:
                r = requests.get(url, headers=headers, params=p, timeout=120)
                if r.status_code == 429:
                    wait = 2 ** attempt + 2
                    log.warning("PJM 429 rate-limited; sleeping %ds", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                payload = r.json()
                break
            except Exception as e:
                wait = 2 ** attempt + 1
                log.warning("PJM request failed (%s). Retrying in %ds", e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"PJM feed {feed} failed after retries")

        items = payload.get("items", [])
        all_rows.extend(items)
        total = int(payload.get("totalRows", len(items)))
        if len(all_rows) >= total or len(items) < ROW_COUNT:
            break
        start_row = len(all_rows) + 1
    return all_rows
