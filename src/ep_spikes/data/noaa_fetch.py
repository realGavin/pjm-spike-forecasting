"""NOAA Storm Events bulk CSV fetch + filter.

Free, no auth. Source: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
"""
from __future__ import annotations

import gzip
import io
import logging
import re
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
DETAILS_PATTERN = re.compile(r'href="(StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d+)\.csv\.gz)"')


def discover_details_file(year: int) -> str:
    """Pick the most recent details CSV for a given year from the NCEI index."""
    r = requests.get(BASE_URL, timeout=30)
    r.raise_for_status()
    matches = [(fname, int(y), int(c)) for fname, y, c in DETAILS_PATTERN.findall(r.text)]
    matches = [m for m in matches if m[1] == year]
    if not matches:
        raise RuntimeError(f"No StormEvents details file found for {year}")
    best = max(matches, key=lambda t: t[2])
    return best[0]


def fetch_year_csv(year: int, *, cache_dir: Path, use_cache: bool = True) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = discover_details_file(year)
    fp = cache_dir / fname
    if use_cache and fp.exists():
        log.info("NOAA cache hit: %s", fname)
        return fp
    url = BASE_URL + fname
    log.info("Downloading NOAA %s", url)
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(fp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 15):
            f.write(chunk)
    return fp


def load_csv(fp: Path) -> pd.DataFrame:
    with gzip.open(fp, "rb") as gz:
        raw = gz.read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def fetch_storm_events(
    start_year: int,
    end_year: int,
    *,
    cache_dir: Path,
    states: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Concat multi-year NOAA storm events, filter by state, parse UTC timestamps."""
    frames: list[pd.DataFrame] = []
    for y in range(start_year, end_year + 1):
        fp = fetch_year_csv(y, cache_dir=cache_dir, use_cache=use_cache)
        df = load_csv(fp)
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)

    if states is not None:
        events["STATE"] = events["STATE"].astype(str).str.upper()
        upper_states = [s.upper() for s in states]
        events = events[events["STATE"].isin(upper_states)].copy()

    events["begin_utc"] = _parse_event_timestamp(
        events["BEGIN_DATE_TIME"], events.get("CZ_TIMEZONE", None)
    )
    events["end_utc"] = _parse_event_timestamp(
        events["END_DATE_TIME"], events.get("CZ_TIMEZONE", None)
    )
    events = events.dropna(subset=["begin_utc"])

    severe_types = {
        "Tornado", "Thunderstorm Wind", "Hail", "Blizzard", "Winter Storm",
        "Ice Storm", "Heat", "Excessive Heat", "Extreme Cold/Wind Chill",
        "Cold/Wind Chill", "High Wind", "Hurricane", "Tropical Storm",
        "Flood", "Flash Flood", "Heavy Snow",
    }
    events["is_severe"] = events["EVENT_TYPE"].isin(severe_types)

    keep = [
        "EVENT_ID", "STATE", "EVENT_TYPE", "CZ_NAME", "CZ_TIMEZONE",
        "BEGIN_DATE_TIME", "END_DATE_TIME", "begin_utc", "end_utc", "is_severe",
    ]
    keep = [c for c in keep if c in events.columns]
    return events[keep].reset_index(drop=True)


_TZ_MAP = {
    "EST": "Etc/GMT+5", "EST-5": "Etc/GMT+5",
    "CST": "Etc/GMT+6", "CST-6": "Etc/GMT+6",
    "MST": "Etc/GMT+7", "MST-7": "Etc/GMT+7",
    "PST": "Etc/GMT+8", "PST-8": "Etc/GMT+8",
    "AKST": "Etc/GMT+9", "AKST-9": "Etc/GMT+9",
    "HST": "Etc/GMT+10", "HST-10": "Etc/GMT+10",
    "EDT": "Etc/GMT+4", "CDT": "Etc/GMT+5",
    "MDT": "Etc/GMT+6", "PDT": "Etc/GMT+7",
}


def _parse_event_timestamp(dt_str: pd.Series, cz_tz: pd.Series | None) -> pd.Series:
    """NOAA timestamps are local-to-reporting-zone. Parse + convert to UTC.

    BEGIN/END_DATE_TIME format is '%d-%b-%y %H:%M:%S' (e.g. '05-MAY-23 14:32:00').
    If we can't resolve a timezone, assume EST (most PJM states are in ET).
    """
    parsed = pd.to_datetime(dt_str, format="%d-%b-%y %H:%M:%S", errors="coerce")
    if cz_tz is None:
        cz_tz = pd.Series(["EST"] * len(parsed))
    tz_resolved = cz_tz.astype(str).str.upper().str.strip().map(_TZ_MAP).fillna("Etc/GMT+5")
    out = pd.Series(pd.NaT, index=parsed.index, dtype="datetime64[ns, UTC]")
    for tz_name, group_idx in tz_resolved.groupby(tz_resolved).groups.items():
        group = parsed.loc[group_idx].dropna()
        if group.empty:
            continue
        localized = group.dt.tz_localize(tz_name, nonexistent="shift_forward", ambiguous="NaT")
        out.loc[group.index] = localized.dt.tz_convert("UTC")
    return out
