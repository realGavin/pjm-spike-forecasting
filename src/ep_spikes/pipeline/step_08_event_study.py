"""Step 8 (Phase 2): event study on first training fold (leakage-safe)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..analysis.event_study import run_event_study
from ..config import load_settings

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    first_fold = settings.folds[0]
    for node in settings.data.nodes:
        fp = paths.PROCESSED_DIR / f"panel_{node.lower()}.parquet"
        if not fp.exists():
            log.warning("Panel missing: %s", fp); continue
        panel = pd.read_parquet(fp)
        out_csv = paths.OUT_TABLES / f"event_study_{node.lower()}.csv"
        run_event_study(
            panel, first_fold.train_start, first_fold.train_end,
            target_col="lmp", out_csv=out_csv,
        )
        log.info("Event study %s written to %s", node, out_csv)


if __name__ == "__main__":
    main()
