"""Step 5: CVaR95 procurement-cost simulation per fold per spike model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..config import load_settings
from ..eval.plots import cvar_curve
from ..risk.cvar import simulate_hedge, daily_cost, var_cvar

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths.ensure_all()

    summary_rows: list[dict] = []
    preds_dir = paths.OUT_PREDS

    for node in settings.data.nodes:
        panel_fp = paths.PROCESSED_DIR / f"panel_{node.lower()}.parquet"
        if not panel_fp.exists():
            log.warning("Panel missing: %s", panel_fp); continue
        panel = pd.read_parquet(panel_fp)
        load_s = panel["metered_load"].astype(float)
        price_s = panel["lmp"].astype(float)

        for fold in settings.folds:
            for pred_fp in preds_dir.glob(f"{node}_{fold.fold_id}_*_label_*.parquet"):
                pred = pd.read_parquet(pred_fp)
                pred.index = pd.DatetimeIndex(pred.index)
                if pred.index.tz is None:
                    pred.index = pred.index.tz_localize("UTC")
                probs = pred["y_prob"]
                common = probs.index.intersection(load_s.index)
                if len(common) == 0:
                    log.warning("No overlap for %s; skip", pred_fp.name)
                    continue
                loadc = load_s.loc[common]
                pricec = price_s.loc[common]
                probsc = probs.loc[common]
                loadc = loadc.fillna(loadc.median())

                tbl = simulate_hedge(
                    loadc, pricec, probsc,
                    thresholds=settings.cvar.hedge_thresholds,
                    forward_price=settings.cvar.forward_price,
                    alpha=settings.cvar.alpha,
                )
                model_tag = pred_fp.stem.replace(f"{node}_{fold.fold_id}_", "")
                tbl.insert(0, "model_tag", model_tag)
                tbl.insert(0, "fold_id", fold.fold_id)
                tbl.insert(0, "node", node)
                summary_rows.append(tbl)

                fig = paths.OUT_FIG / f"{node}_{fold.fold_id}_{model_tag}_cvar.png"
                ths = tbl[tbl["strategy"] != "no_hedge"]["threshold"].tolist()
                cvs = tbl[tbl["strategy"] != "no_hedge"]["cvar"].tolist()
                base = float(tbl[tbl["strategy"] == "no_hedge"]["cvar"].iloc[0])
                cvar_curve(ths, cvs, base, f"CVaR95 {node} {fold.fold_id} {model_tag}", fig)

    if not summary_rows:
        log.warning("No CVaR results written.")
        return
    big = pd.concat(summary_rows, ignore_index=True)
    out = paths.OUT_TABLES / "cvar_summary.csv"
    big.to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
