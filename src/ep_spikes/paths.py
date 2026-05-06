"""Canonical project paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PJM = RAW_DIR / "pjm"
RAW_WEATHER = RAW_DIR / "weather"
RAW_NOAA = RAW_DIR / "noaa"

OUTPUTS = ROOT / "outputs"
OUT_MODELS = OUTPUTS / "models"
OUT_PREDS = OUTPUTS / "predictions"
OUT_FIG = OUTPUTS / "figures"
OUT_TABLES = OUTPUTS / "tables"
OUT_REPORTS = OUTPUTS / "reports"


def ensure_all() -> None:
    for p in [
        RAW_PJM, RAW_WEATHER, RAW_NOAA, INTERIM_DIR, PROCESSED_DIR,
        OUT_MODELS, OUT_PREDS, OUT_FIG, OUT_TABLES, OUT_REPORTS,
    ]:
        p.mkdir(parents=True, exist_ok=True)
