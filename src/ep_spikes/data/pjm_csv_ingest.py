"""Ingest manually-downloaded DataMiner2 CSVs and convert them to the same
per-year parquet format the REST fetchers use.

Expected source filenames in ``data/raw/pjm_csv/``:
    da_hrl_lmps_{zone}_{YYYY}.csv
    hrl_load_metered_{zone}_{YYYY}.csv
    load_frcstd_hist_{zone}_{YYYY}.csv   (optional)

The column names emitted by DataMiner2's web UI are normalized here — we accept
either the snake_case JSON schema or the Title Case UI schema.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and collapse spaces to underscores."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df


def _to_utc(local_series: pd.Series, src_tz: str = "America/New_York") -> pd.DatetimeIndex:
    local = pd.to_datetime(local_series, errors="coerce")
    utc = local.dt.tz_localize(
        src_tz, nonexistent="shift_forward", ambiguous="infer"
    ).dt.tz_convert("UTC")
    return pd.DatetimeIndex(utc)


def ingest_da_lmp_csv(csv_path: Path, out_parquet: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(csv_path))
    ts_col = next((c for c in ("datetime_beginning_ept", "datetime_beginning_utc")
                   if c in df.columns), None)
    if ts_col is None:
        raise KeyError(f"Could not find EPT/UTC timestamp column in {csv_path.name}")
    idx = _to_utc(df[ts_col]) if "ept" in ts_col else pd.DatetimeIndex(
        pd.to_datetime(df[ts_col], utc=True)
    )
    out = pd.DataFrame(index=idx)
    out.index.name = "timestamp_utc"
    for src, dst in [
        ("total_lmp_da", "lmp"),
        ("congestion_price_da", "congestion_lmp"),
        ("marginal_loss_price_da", "loss_lmp"),
    ]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    if "pnode_name" in df.columns:
        out["zone"] = df["pnode_name"].astype(str).to_numpy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet)
    log.info("Ingested %s -> %s (%d rows)", csv_path.name, out_parquet.name, len(out))
    return out


def ingest_metered_load_csv(csv_path: Path, out_parquet: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(csv_path))
    ts_col = next((c for c in ("datetime_beginning_ept", "datetime_beginning_utc")
                   if c in df.columns), None)
    if ts_col is None:
        raise KeyError(f"Could not find EPT/UTC timestamp column in {csv_path.name}")
    idx = _to_utc(df[ts_col]) if "ept" in ts_col else pd.DatetimeIndex(
        pd.to_datetime(df[ts_col], utc=True)
    )
    out = pd.DataFrame(index=idx)
    out.index.name = "timestamp_utc"
    load_col = next((c for c in ("mw", "area_load_mwh", "load_mwh") if c in df.columns), None)
    if load_col is None:
        raise KeyError(f"No load column (mw / area_load_mwh) in {csv_path.name}")
    out["load_mw"] = pd.to_numeric(df[load_col], errors="coerce")
    if "load_area" in df.columns:
        out["zone"] = df["load_area"].astype(str).to_numpy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet)
    log.info("Ingested %s -> %s (%d rows)", csv_path.name, out_parquet.name, len(out))
    return out


def ingest_load_forecast_csv(csv_path: Path, out_parquet: Path, zone: str) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(csv_path))
    target_col = next(
        (c for c in ("forecast_datetime_beginning_ept", "forecast_hour_beginning_ept")
         if c in df.columns), None,
    )
    issue_col = next(
        (c for c in ("evaluated_at_ept", "forecast_creation_datetime_ept") if c in df.columns),
        None,
    )
    mw_col = next((c for c in ("forecast_load_mw", "forecast_mw") if c in df.columns), None)
    if not (target_col and issue_col and mw_col):
        raise KeyError(f"Missing expected columns in {csv_path.name} (got {list(df.columns)})")
    target_utc = _to_utc(df[target_col])
    issue_utc = _to_utc(df[issue_col])
    out = pd.DataFrame({
        "target_utc": target_utc,
        "issue_utc": issue_utc,
        "forecast_mw": pd.to_numeric(df[mw_col], errors="coerce"),
        "zone": zone,
    })
    out = out.dropna(subset=["target_utc", "issue_utc", "forecast_mw"])
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet)
    log.info("Ingested %s -> %s (%d rows)", csv_path.name, out_parquet.name, len(out))
    return out


def ingest_all(csv_dir: Path, parquet_dir: Path, zone: str) -> None:
    zlow = zone.lower()
    for csv in sorted(csv_dir.glob(f"da_hrl_lmps_{zlow}_*.csv")):
        year = re.search(r"_(\d{4})", csv.name).group(1)
        ingest_da_lmp_csv(csv, parquet_dir / f"da_lmp_{zlow}_{year}.parquet")
    for csv in sorted(csv_dir.glob(f"hrl_load_metered_{zlow}_*.csv")):
        year = re.search(r"_(\d{4})", csv.name).group(1)
        ingest_metered_load_csv(csv, parquet_dir / f"metered_load_{zlow}_{year}.parquet")
    for csv in sorted(csv_dir.glob(f"load_frcstd_hist_{zlow}_*.csv")):
        year = re.search(r"_(\d{4})", csv.name).group(1)
        ingest_load_forecast_csv(
            csv, parquet_dir / f"da_load_forecast_{zlow}_{year}.parquet", zone=zone,
        )


if __name__ == "__main__":
    import argparse
    from .. import paths
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="COMED")
    parser.add_argument("--csv-dir", type=Path, default=paths.RAW_DIR / "pjm_csv")
    parser.add_argument("--out-dir", type=Path, default=paths.RAW_PJM)
    args = parser.parse_args()
    ingest_all(args.csv_dir, args.out_dir, args.zone)
