"""Step 4: rolling-origin train+eval for spike and quantile tasks."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..config import load_settings
from ..eval.rolling import run_quantile_fold, run_spike_fold

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", default="label_either",
                        choices=["label_abs", "label_rel", "label_either"])
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths.ensure_all()

    results_rows: list[dict] = []
    for node in settings.data.nodes:
        fp = paths.PROCESSED_DIR / f"panel_{node.lower()}.parquet"
        if not fp.exists():
            log.warning("Panel missing: %s; skipping %s", fp, node)
            continue
        panel = pd.read_parquet(fp)
        panel = panel.dropna(subset=["lmp"])

        for fold in settings.folds:
            spike_results = run_spike_fold(
                panel.copy(), fold, node, settings, label_col=args.label,
                out_preds_dir=paths.OUT_PREDS, out_fig_dir=paths.OUT_FIG,
                extra_models=settings.models.extra_models,
            )
            for r in spike_results:
                row = {"fold_id": r.fold_id, "node": r.node, "model": r.model_name,
                       "task": r.task, "label": r.label, "best_params": json.dumps(r.best_params, default=str)}
                row.update(r.metrics)
                row["preds_path"] = str(r.preds_path)
                results_rows.append(row)

            q_result = run_quantile_fold(
                panel.copy(), fold, node, settings,
                out_preds_dir=paths.OUT_PREDS,
            )
            row = {"fold_id": q_result.fold_id, "node": q_result.node, "model": q_result.model_name,
                   "task": q_result.task, "label": q_result.label,
                   "best_params": json.dumps(q_result.best_params, default=str)}
            row.update(q_result.metrics)
            row["preds_path"] = str(q_result.preds_path)
            results_rows.append(row)

    tbl = pd.DataFrame(results_rows)
    out = paths.OUT_TABLES / "results_mvp.csv"
    tbl.to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(tbl))


if __name__ == "__main__":
    main()
