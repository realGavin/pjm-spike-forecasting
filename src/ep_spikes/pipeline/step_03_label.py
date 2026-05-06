"""Step 3: sanity-report label prevalence.

Per-fold seasonal thresholds are computed inside run_spike_fold.  This step
just reports prevalence with a *dataset-wide* threshold for EDA only; it does
NOT write labels back to the panel for training.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..config import load_settings
from ..labels.spike import compute_seasonal_thresholds, make_labels

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths.ensure_all()

    rows: list[dict] = []
    for node in settings.data.nodes:
        fp = paths.PROCESSED_DIR / f"panel_{node.lower()}.parquet"
        if not fp.exists():
            log.warning("Panel missing: %s", fp)
            continue
        panel = pd.read_parquet(fp)
        seasons = panel["season_name"].astype(str)
        thr = compute_seasonal_thresholds(panel["lmp"], seasons, q=settings.labels.seasonal_quantile)
        labels = make_labels(
            panel["lmp"], seasons,
            absolute_threshold=settings.labels.absolute_threshold,
            seasonal_thresholds=thr,
        )
        rows.append({
            "node": node,
            "n_rows": len(panel),
            "absolute_threshold": settings.labels.absolute_threshold,
            "summer_q95": thr.get("summer"),
            "winter_q95": thr.get("winter"),
            "shoulder_q95": thr.get("shoulder"),
            "prev_abs": float(labels["label_abs"].mean()),
            "prev_rel": float(labels["label_rel"].mean()),
            "prev_either": float(labels["label_either"].mean()),
        })
        log.info("Node %s: abs=%.3f%%  rel=%.3f%%  either=%.3f%%",
                 node, 100 * labels["label_abs"].mean(),
                 100 * labels["label_rel"].mean(),
                 100 * labels["label_either"].mean())

    out = paths.OUT_TABLES / "label_prevalence_overview.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
