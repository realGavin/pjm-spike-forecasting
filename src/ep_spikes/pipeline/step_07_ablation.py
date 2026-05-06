"""Step 7 (Phase 2): ablation study — market-only vs market+weather vs full feature sets."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..config import load_settings
from ..eval.rolling import run_spike_fold

log = logging.getLogger(__name__)

ABLATION_SETS: dict[str, list[str]] = {
    "market_only": ["market"],
    "market_plus_weather": ["market", "weather_only"],
    "market_plus_storms": ["market", "storm_only"],
    "full": ["all"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", default="label_either")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows: list[dict] = []
    for node in settings.data.nodes:
        fp = paths.PROCESSED_DIR / f"panel_{node.lower()}.parquet"
        if not fp.exists():
            log.warning("Panel missing: %s; skip %s", fp, node)
            continue
        panel = pd.read_parquet(fp).dropna(subset=["lmp"])
        for fold in settings.folds:
            for tag, sets in ABLATION_SETS.items():
                log.info("Ablation %s / %s / %s", node, fold.fold_id, tag)
                results = run_spike_fold(
                    panel.copy(), fold, node, settings, label_col=args.label,
                    out_preds_dir=paths.OUT_PREDS,
                    out_fig_dir=paths.OUT_FIG,
                    extra_models=[],
                    feature_sets=sets,
                    tag_suffix=f"abl_{tag}",
                )
                for r in results:
                    rows.append({
                        "fold_id": fold.fold_id, "ablation": tag, "node": r.node,
                        "model": r.model_name, "label": r.label,
                        **r.metrics,
                        "best_params": json.dumps(r.best_params, default=str),
                    })

    out = paths.OUT_TABLES / "ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
