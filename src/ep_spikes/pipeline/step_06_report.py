"""Step 6: render a simple markdown report from results tables."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..config import load_settings

log = logging.getLogger(__name__)


def _format_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    results_fp = paths.OUT_TABLES / "results_mvp.csv"
    cvar_fp = paths.OUT_TABLES / "cvar_summary.csv"
    prev_fp = paths.OUT_TABLES / "label_prevalence_overview.csv"

    parts: list[str] = []
    parts.append("# PJM Day-Ahead Spike Forecasting -- MVP Results\n")
    parts.append(f"Config: `{args.config.name}`\n")
    parts.append(f"Nodes: {', '.join(settings.data.nodes)}\n")
    parts.append(f"Window: {settings.data.start_date} to {settings.data.end_date}\n")

    if prev_fp.exists():
        parts.append("\n## Label prevalence (dataset-wide)\n")
        parts.append(_format_table(pd.read_csv(prev_fp)))

    if results_fp.exists():
        res = pd.read_csv(results_fp)
        parts.append("\n## Model metrics per fold\n")
        keep = [c for c in ["fold_id","node","model","task","label","prevalence","aucpr","aucroc","brier","lift_over_prevalence","pinball_q95","pinball_q99","coverage_q95","coverage_q99"] if c in res.columns]
        parts.append(_format_table(res[keep]))

    if cvar_fp.exists():
        cv = pd.read_csv(cvar_fp)
        parts.append("\n## CVaR95 of daily procurement cost\n")
        keep = [c for c in ["node","fold_id","model_tag","strategy","threshold","mean_daily_cost","var","cvar","n_days","n_triggered_hours"] if c in cv.columns]
        parts.append(_format_table(cv[keep]))

    parts.append("\n## Known limitations (MVP)\n")
    parts.append("- Weather proxy: Open-Meteo archive = realized obs, not as-of-origin forecasts. "
                 "Biases AUCPR upward. Phase 2 swaps in Meteomatics issued forecasts.\n"
                 "- Quantile crossing: P95>P99 corrected post-hoc via monotone broadcast.\n"
                 "- Phase 2 adds LSTM/TCN + LEAR benchmarks, PECO node, and a leakage-audited event study.\n")

    out = paths.OUT_REPORTS / "mvp_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
