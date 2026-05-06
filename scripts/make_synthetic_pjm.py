"""Generate synthetic PJM DA LMP + metered load + DA load forecast parquet files.

Used for smoke-testing the pipeline while waiting for a real PJM_API_KEY.
The synthetic data lives in data/raw/pjm/ using the same filenames the real
fetchers write, so they are picked up by the panel builder as cache hits.

Run:
    python scripts/make_synthetic_pjm.py --year 2023 --zone COMED --out data/raw/pjm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _hourly_utc_index(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    return pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz="UTC")


def synthesize_da_lmp(year: int, zone: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year)
    idx = _hourly_utc_index(year)
    local = idx.tz_convert("America/New_York")
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()

    base = 35 + 8 * np.sin(2 * np.pi * (hour - 16) / 24) + 5 * (dow < 5).astype(float)
    summer_boost = np.where(np.isin(month, [6, 7, 8]), 12.0, 0.0)
    winter_boost = np.where(np.isin(month, [12, 1, 2]), 6.0, 0.0)
    noise = rng.normal(0, 8, len(idx))
    lmp = base + summer_boost + winter_boost + noise

    # Inject summer heat-event spikes
    for _ in range(20):
        ts_idx = rng.integers(0, len(idx))
        if month[ts_idx] in (6, 7, 8) and hour[ts_idx] in range(14, 21):
            spike_len = rng.integers(2, 6)
            spike_mag = rng.uniform(250, 900)
            end_idx = min(ts_idx + spike_len, len(idx))
            lmp[ts_idx:end_idx] += spike_mag
    # Inject winter cold-snap spikes
    for _ in range(10):
        ts_idx = rng.integers(0, len(idx))
        if month[ts_idx] in (12, 1, 2) and hour[ts_idx] in (7, 8, 18, 19, 20):
            spike_len = rng.integers(2, 5)
            spike_mag = rng.uniform(300, 1200)
            end_idx = min(ts_idx + spike_len, len(idx))
            lmp[ts_idx:end_idx] += spike_mag

    lmp = np.clip(lmp, 5.0, 5000.0)
    df = pd.DataFrame({
        "lmp": lmp,
        "congestion_lmp": rng.normal(0, 2, len(idx)),
        "loss_lmp": rng.normal(1, 0.5, len(idx)),
        "zone": zone,
    }, index=idx)
    df.index.name = "timestamp_utc"
    return df


def synthesize_metered_load(year: int, zone: str, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year)
    idx = _hourly_utc_index(year)
    local = idx.tz_convert("America/New_York")
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()

    base = 12000 + 3500 * np.sin(2 * np.pi * (hour - 16) / 24)
    weekend_drop = np.where(dow >= 5, -800, 0)
    summer_peak = np.where(np.isin(month, [6, 7, 8]), 4500, 0)
    winter_peak = np.where(np.isin(month, [12, 1, 2]), 2500, 0)
    noise = rng.normal(0, 300, len(idx))
    load_mw = np.clip(base + weekend_drop + summer_peak + winter_peak + noise, 5000, 22000)

    df = pd.DataFrame({"load_mw": load_mw, "zone": zone}, index=idx)
    df.index.name = "timestamp_utc"
    return df


def synthesize_da_load_forecast(year: int, zone: str, seed: int = 44) -> pd.DataFrame:
    """Simulate daily forecast issued at ~09:00 EPT on d-1 for all 24 hrs of day d."""
    rng = np.random.default_rng(seed + year)
    idx = _hourly_utc_index(year)
    metered = synthesize_metered_load(year, zone, seed=seed + 100)["load_mw"]
    noise = rng.normal(0, 200, len(idx))
    fcst = (metered + noise).clip(5000, 22000)

    local = idx.tz_convert("America/New_York")
    days = pd.Index(local.date).unique()

    rows: list[dict] = []
    for d in days:
        target_local_start = pd.Timestamp(d, tz="America/New_York")
        target_local_end = target_local_start + pd.Timedelta(days=1)
        day_mask = (local >= target_local_start) & (local < target_local_end)
        if not day_mask.any():
            continue
        prev_day = target_local_start - pd.Timedelta(days=1)
        issue_local = pd.Timestamp(
            year=prev_day.year, month=prev_day.month, day=prev_day.day,
            hour=9, tz="America/New_York",
        )
        issue_utc = issue_local.tz_convert("UTC")
        for ts in idx[day_mask]:
            rows.append({
                "target_utc": ts,
                "issue_utc": issue_utc,
                "forecast_mw": float(fcst.loc[ts]),
                "zone": zone,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2023])
    parser.add_argument("--zone", default="COMED")
    parser.add_argument("--out", type=Path, default=Path("data/raw/pjm"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        lmp = synthesize_da_lmp(year, args.zone)
        lmp.to_parquet(args.out / f"da_lmp_{args.zone.lower()}_{year}.parquet")
        ld = synthesize_metered_load(year, args.zone)
        ld.to_parquet(args.out / f"metered_load_{args.zone.lower()}_{year}.parquet")
        fc = synthesize_da_load_forecast(year, args.zone)
        fc.to_parquet(args.out / f"da_load_forecast_{args.zone.lower()}_{year}.parquet")
        print(f"Wrote synthetic PJM data for {args.zone} {year}: {len(lmp)} price rows, "
              f"{len(ld)} load rows, {len(fc)} fcst rows")


if __name__ == "__main__":
    main()
