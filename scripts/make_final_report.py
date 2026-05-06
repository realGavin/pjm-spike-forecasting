"""Generate the final project report (.docx) from artifacts in outputs/.

Pulls from:
  outputs/tables/results_mvp.csv        (per-fold per-model metrics)
  outputs/tables/cvar_summary.csv       (CVaR hedge sim)
  outputs/tables/ablation_results.csv   (feature-set contributions)
  outputs/tables/event_study_<zone>.csv (storm coefficients)
  outputs/tables/label_prevalence_overview.csv
  outputs/figures/*.png                 (CVaR curves, reliability)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

ROOT = Path("/Users/xujialin/Desktop/energy/Predicting_Electricity_Price_Spikes")
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
OUT = Path("/Users/xujialin/Desktop/IND_ENG_290_Final_Report.docx")


def _shade(cell, color: str = "D9E1F2") -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_p(doc: Document, text: str, *, bold=False, italic=False, size: int | None = None,
          align: str | None = None) -> None:
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size is not None:
        r.font.size = Pt(size)


def add_heading(doc: Document, text: str, level: int) -> None:
    doc.add_heading(text, level=level)


def add_table(doc: Document, header: list[str], rows: list[list[str]]) -> None:
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = "Light Grid Accent 1"
    for j, h in enumerate(header):
        c = tbl.rows[0].cells[j]; c.text = h
        for run in c.paragraphs[0].runs:
            run.bold = True; run.font.size = Pt(10)
        _shade(c)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = tbl.rows[i].cells[j]; c.text = str(v)
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    for r in tbl.rows:
        for c in r.cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_bullets(doc: Document, items: list[str]) -> None:
    for t in items:
        doc.add_paragraph(t, style="List Bullet")


def _safe_read(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def _add_image(doc: Document, path: Path, caption: str, width_in: float = 5.5) -> None:
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width_in))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption); r.italic = True; r.font.size = Pt(9)


def build() -> None:
    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Calibri"; style.font.size = Pt(11)
    for section in doc.sections:
        section.top_margin = Cm(2.0); section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2); section.right_margin = Cm(2.2)

    # ============ Title page ============
    add_p(doc, "INDENG 290 — Energy Analytics", bold=True, size=18, align="center")
    add_p(doc, "Final Project Report", bold=True, size=16, align="center")
    doc.add_paragraph()
    add_p(doc,
          "Predicting Electricity Price Spikes from Extreme Weather in PJM:\n"
          "Deep Learning Forecasts and Cost-at-Risk Quantification",
          bold=True, italic=True, size=14, align="center")
    doc.add_paragraph()
    add_p(doc, "Team: Gavin Zeng, Alex Yu, Jialin Xu, Andrew Lai", align="center", size=12)
    add_p(doc, "Course: INDENG 290 (Energy Analytics)", align="center", size=12)
    add_p(doc, "Submission date: May 11, 2026", align="center", size=12)
    doc.add_paragraph()

    # ============ Abstract ============
    add_heading(doc, "Abstract", level=1)
    add_p(doc,
          "Tail price events in PJM's day-ahead electricity market are rare but responsible "
          "for a disproportionate share of procurement cost for load-serving entities. This "
          "project builds an end-to-end machine-learning toolkit that (i) links PJM "
          "day-ahead LMPs with real hourly weather and NOAA storm-event indicators under a "
          "strict forecast-origin information cutoff, (ii) predicts both spike probability "
          "(binary) and upper quantiles (P95 / P99) of the next-day price distribution using "
          "a stack of baselines (logistic regression, XGBoost, LEAR) and deep-learning models "
          "(MLP and optional LSTM), and (iii) translates those predictions into a "
          "Cost-at-Risk (CVaR95) decision metric for a stylized LSE. We evaluate on a "
          "rolling-origin protocol over 2019-2024 with 3-year training windows and 1-year "
          "test windows. XGBoost with the full feature set achieves out-of-sample AUCPR "
          "substantially above prevalence; weather and storm features contribute "
          "independently but modestly relative to market-lag features; and a simple "
          "probability-triggered hedge reduces CVaR95 in stress years. A leakage-audited "
          "event study confirms that excessive-heat and high-wind NOAA events are "
          "significant predictors of log-price, consistent with grid-scarcity intuition.")

    # ============ 1. Introduction ============
    add_heading(doc, "1. Introduction and Motivation", level=1)
    add_p(doc,
          "Wholesale electricity prices are typically calm — COMED day-ahead LMPs average "
          "about $30/MWh in our training window — but severe weather can push the system into "
          "scarcity and drive prices above $1,000/MWh for contiguous hours. These tail events "
          "are rare (roughly 1% of hours in stress years) but drive a disproportionate share "
          "of annual procurement cost for a load-serving entity (LSE) that buys on the "
          "day-ahead market. Standard forecasting models underpredict them, leaving LSEs "
          "poorly hedged.")
    add_p(doc,
          "We ask: can explicit extreme-weather signals combined with modern ML models meaningfully "
          "improve day-ahead tail-risk forecasts in PJM, and does that improvement translate "
          "into measurable Cost-at-Risk reduction for a stylized LSE client?")

    add_heading(doc, "1.1 Research questions", level=2)
    add_bullets(doc, [
        "How accurately can we predict next-day price-spike probability and upper-quantile "
        "prices (P95 / P99) in PJM under a forecast-origin-compliant information set?",
        "How much do explicit extreme-weather and storm-event features improve tail-risk "
        "prediction over market-only models?",
        "Do richer models (XGBoost, LEAR, MLP, LSTM) outperform a logistic-regression baseline "
        "on spike detection?",
        "How much can improved tail prediction reduce daily Cost-at-Risk (CVaR95) for a "
        "stylized LSE that hedges into a DA-forward contract when predicted spike probability "
        "crosses a threshold?",
    ])

    add_heading(doc, "1.2 Stylized client", level=2)
    add_p(doc,
          "Our client is an LSE procuring power in PJM's day-ahead market on behalf of "
          "retail customers. Hourly procurement cost is C_t = L_t × P_t where L_t is "
          "zonal metered load and P_t is the DA clearing price. The client cares about the "
          "right tail of daily procurement cost — specifically CVaR95 of ΣL·P — because "
          "extreme-cost days most threaten margins. Improved spike forecasts let the LSE "
          "pre-purchase energy in forward markets or activate demand response before prices "
          "surge.")

    # ============ 2. Market context ============
    add_heading(doc, "2. Market and Node Selection", level=1)
    add_p(doc,
          "PJM is the largest U.S. wholesale electricity market. Day-ahead (DA) LMPs clear "
          "by midday d-1 for all 24 hours of day d. Our primary pricing node is COMED "
          "(Commonwealth Edison, Chicago area); as a secondary comparison we include PECO "
          "(Philadelphia area), geographically distant on the bulk power system, to examine "
          "whether transmission congestion and regional weather regimes produce meaningfully "
          "different spike patterns.")
    add_p(doc,
          "We target the forecast origin at 10:00 AM EPT on day d-1, aligned with PJM's "
          "DA market which closes at noon on d-1. At this origin the model has access to "
          "(1) all settled DA LMPs through day d-1 (cleared the prior afternoon), (2) PJM's "
          "published DA load forecast for day d, (3) day-ahead weather forecasts for day d, "
          "and (4) NOAA storm events whose start timestamps are strictly before the origin.")

    # ============ 3. Data ============
    add_heading(doc, "3. Data", level=1)
    add_table(doc, ["Source", "Use", "Window", "Notes"], [
        ["PJM DataMiner 2 — da_hrl_lmps", "DA LMP target + lags", "2019-2024",
         "Zonal LMP for COMED & PECO via DataMiner 2 API"],
        ["PJM DataMiner 2 — hrl_load_metered", "Load exposure for CaR", "2019-2024",
         "COMED = 'CE' and PECO = 'PE' codes"],
        ["PJM DataMiner 2 — load_frcstd_hist", "As-of-origin DA load forecast", "2019-2024",
         "Used with issue_utc ≤ origin as the vintage filter"],
        ["Open-Meteo ERA5 archive", "Weather features", "2019-2024",
         "Free, no-auth; temperature, humidity, precip, wind, gusts at Chicago & Philadelphia"],
        ["NOAA Storm Events DB", "Extreme-weather indicators", "2019-2024",
         "Free bulk CSV; filtered to 14 PJM-footprint states"],
    ])
    add_p(doc,
          "We use realized Open-Meteo ERA5 observations as a proxy for day-ahead weather "
          "forecasts in this project. This biases tail-risk metrics upward because a perfect "
          "weather forecast is assumed; we discuss the bias and the path to replacing the "
          "proxy with Meteomatics issued forecasts in Section 10.")

    # ============ Price visualisation (context) ============
    for node in ["comed", "peco"]:
        ts = FIGS / f"lmp_timeseries_{node}.png"
        if ts.exists():
            _add_image(doc, ts, f"Figure: {node.upper()} DA LMP daily mean and daily max (2019–2024)",
                       width_in=6.0)
        ds = FIGS / f"lmp_distribution_{node}.png"
        if ds.exists():
            _add_image(doc, ds, f"Figure: {node.upper()} DA LMP distribution and seasonal density",
                       width_in=6.0)

    # ============ 4. Methodology ============
    add_heading(doc, "4. Methodology", level=1)

    add_heading(doc, "4.1 Forecast-origin information set", level=2)
    add_p(doc,
          "Every feature builder receives an explicit origin timestamp and is required to "
          "use only pre-origin data. We enforce this programmatically in code: a dedicated "
          "pytest fixture corrupts post-origin values in the source series and asserts that "
          "derived features do not change. PJM's DA market structure makes the information "
          "set self-consistent: day d-1's 24 hours of DA LMPs clear early afternoon on "
          "d-2, so all of them are available at 10:00 AM d-1.")

    add_heading(doc, "4.2 Feature engineering", level=2)
    add_bullets(doc, [
        "Calendar — hour of day, day of week, month, season, US federal holiday, cyclic "
        "sin/cos encodings of hour and day-of-year.",
        "Prices — lags at 24 / 48 / 168 h relative to each target hour; rolling mean, std, "
        "max, min over 24 h and 168 h, closing at origin − 1 s; 24-hour price ramp.",
        "Load — published DA forecast filtered to issue_utc ≤ origin (as-of join), lagged "
        "metered load through d-2, 24 h rolling mean / max, forecast-to-last ratio.",
        "Weather — hourly temperature, apparent temperature, relative humidity, "
        "precipitation, wind speed, gusts; daily aggregates (max / min / range, total "
        "precip, peak wind); temperature anomaly Z-score vs. a self-built climatology.",
        "Storms — counts of NOAA events whose begin_utc < origin in 12 h / 24 h / 48 h "
        "windows, plus counts by event type (Tornado, Winter Storm, Ice Storm, Heat, "
        "Excessive Heat, High Wind).",
    ])

    add_heading(doc, "4.3 Spike labels", level=2)
    add_p(doc,
          "Three binary labels aligned to each hour:")
    add_table(doc, ["Label", "Definition"], [
        ["label_abs", "LMP > $300/MWh (absolute scarcity threshold)"],
        ["label_rel", "LMP in top 5% within its season (summer / winter / shoulder)"],
        ["label_either", "Logical OR of the two — primary training target"],
    ])
    add_p(doc,
          "The seasonal threshold is re-fit on each fold's training window only, never on "
          "the full panel; this is essential to avoid leaking information from the held-out "
          "test year into the label itself.")

    add_heading(doc, "4.4 Rolling-origin evaluation", level=2)
    add_p(doc,
          "We use three folds with fixed 3-year training windows and 1-year test windows: "
          "train 2019–2021 → test 2022; train 2020–2022 → test 2023; train 2021–2023 → "
          "test 2024. Hyperparameter tuning uses TimeSeriesSplit(4) inside each training "
          "window; no test-window data enters tuning.")

    add_heading(doc, "4.5 Models", level=2)
    add_table(doc, ["Family", "Model", "Objective", "Notes"], [
        ["Baseline",  "Logistic regression", "cross-entropy",
         "Median-impute → standardize → LR with class_weight=balanced"],
        ["Tree",      "XGBoost classifier", "cross-entropy",
         "Small grid (n_est, depth, lr); scale_pos_weight auto-set; AUCPR selection"],
        ["Tree",      "XGBoost quantile",  "pinball (P95, P99)",
         "Multi-alpha reg:quantileerror; post-hoc monotone correction"],
        ["Benchmark", "LEAR (Lasso AR + calibrator)", "cross-entropy",
         "LassoCV on the full feature panel; isotonic-calibrated logistic head"],
        ["Deep",      "MLP (fully-connected)", "cross-entropy",
         "sklearn MLPClassifier, (128, 64) hidden, early stopping"],
        ["Deep",      "LSTM (sequence)", "cross-entropy",
         "torch LSTM over 24-hour sliding windows (optional install)"],
    ])

    add_heading(doc, "4.6 Cost-at-Risk simulation", level=2)
    add_p(doc,
          "For each test day d, the LSE's procurement cost is C_d = Σ_h L_h × P_h. We "
          "compute VaR and CVaR at α = 0.95 over the empirical distribution of {C_d} in the "
          "test year. The hedge rule: if predicted spike probability ≥ threshold t for hour "
          "h, we treat the effective price in that hour as a fixed DA-forward "
          "strike (stylized placeholder $65/MWh); otherwise we pay the realized LMP. We "
          "sweep t over {0.05, 0.10, 0.20, 0.30, 0.50} and report CVaR95 per strategy.")

    # ============ 5. Event study ============
    add_heading(doc, "5. Exploratory Event Study", level=1)
    add_p(doc,
          "We run an OLS regression of log(1 + LMP) on storm-event indicators, temperature "
          "features, and hour / day-of-week / month fixed effects, using only the first "
          "fold's training window (2019-2021, or an equivalently-sized window in the "
          "smoke config). Heteroskedasticity-robust standard errors (HC1). This serves as "
          "motivating EDA, not feature selection.")

    ev = _safe_read(TABLES / "event_study_comed.csv")
    if ev is not None and not ev.empty:
        sig = ev[ev["is_event"] == True].sort_values("p").head(10)
        rows = [[str(r["variable"]), f"{r['coef']:+.4f}", f"{r['std_err']:.4f}",
                 f"{r['t']:.2f}", f"{r['p']:.4f}"] for _, r in sig.iterrows()]
        add_table(doc, ["Variable", "Coef", "Std. Err.", "t", "p"], rows)

    # ============ 6. Spike classification results ============
    add_heading(doc, "6. Spike-Probability Classification Results", level=1)
    res = _safe_read(TABLES / "results_mvp.csv")
    if res is not None and not res.empty:
        spike = res[res["task"] == "spike"].copy()
        if len(spike):
            keep_cols = [c for c in ["node","fold_id","model","prevalence","aucpr","aucroc","brier","lift_over_prevalence"] if c in spike.columns]
            spike_fmt = spike[keep_cols].copy()
            for col in ["prevalence","aucpr","aucroc","brier","lift_over_prevalence"]:
                if col in spike_fmt.columns:
                    spike_fmt[col] = spike_fmt[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            add_table(doc, list(spike_fmt.columns),
                      [list(map(str, row)) for row in spike_fmt.to_numpy()])
    else:
        add_p(doc, "[Results pending — will be populated by the full three-fold run on real data.]",
              italic=True)

    aucpr_png = FIGS / "aucpr_by_fold.png"
    if aucpr_png.exists():
        _add_image(doc, aucpr_png, "Figure: AUCPR by fold and model (held-out test year)",
                   width_in=6.2)

    add_heading(doc, "6.1 Reliability of predicted probabilities", level=2)
    for fig in sorted(FIGS.glob("*_reliability.csv"))[:2]:
        png = fig.with_suffix(".png")
        _add_image(doc, png, f"Reliability: {png.stem}", width_in=5.0)

    add_heading(doc, "6.2 Fold-by-fold narrative", level=2)
    add_p(doc,
          "Each of the three rolling-origin folds covers a distinct market regime, which "
          "helps interpret the spread of AUCPR across folds:")
    add_bullets(doc, [
        "Fold 1 (train 2019-2021 → test 2022): 2022 was a notably hot summer in PJM's "
        "western footprint; summer DA LMPs at COMED frequently breached $200/MWh. Models "
        "trained through 2021 get a genuinely out-of-distribution test on 2022 heat events, "
        "making this the hardest fold and the most informative for tail-risk generalization.",
        "Fold 2 (train 2020-2022 → test 2023): 2023 was a relatively mild year at COMED — "
        "we observe zero hours above $300 in our sample. AUCPR here is high because the "
        "seasonal top-5% label is well-separated from the bulk of prices, but the absolute "
        "($300) label is trivially zero-prevalence; the 'either' label is driven entirely "
        "by the seasonal definition.",
        "Fold 3 (train 2021-2023 → test 2024): 2024 reintroduces moderate scarcity events; "
        "in our data we see a mix of summer heat pricing and some winter stress days in "
        "early 2024.",
    ])

    # ============ 7. Quantile results ============
    add_heading(doc, "7. Upper-Quantile Forecasts (P95, P99)", level=1)
    if res is not None:
        q = res[res["task"] == "quantile"].copy()
        if len(q):
            keep = [c for c in ["node","fold_id","pinball_q95","pinball_q99","coverage_q95","coverage_q99"] if c in q.columns]
            qfmt = q[keep].copy()
            for col in ["pinball_q95","pinball_q99","coverage_q95","coverage_q99"]:
                if col in qfmt.columns:
                    qfmt[col] = qfmt[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            add_table(doc, list(qfmt.columns), [list(map(str, r)) for r in qfmt.to_numpy()])
    add_p(doc,
          "Coverage is interpreted as the fraction of test-hour realized LMPs that fall "
          "below the predicted quantile. For a well-calibrated P95 this should be ≈ 0.95; "
          "for P99, ≈ 0.99. The monotone post-hoc correction ensures P99 ≥ P95 per row.")

    # ============ 8. CVaR ============
    add_heading(doc, "8. Cost-at-Risk (CVaR95) Simulation", level=1)
    cv = _safe_read(TABLES / "cvar_summary.csv")
    if cv is not None and not cv.empty:
        sub = cv.copy()
        keep = [c for c in ["node","fold_id","model_tag","strategy","threshold","mean_daily_cost","cvar","n_triggered_hours"] if c in sub.columns]
        sub = sub[keep]
        for col in ["mean_daily_cost","cvar"]:
            if col in sub.columns:
                sub[col] = sub[col].map(lambda v: f"${v/1e6:.2f}M" if pd.notna(v) else "—")
        if "threshold" in sub.columns:
            sub["threshold"] = sub["threshold"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        add_table(doc, list(sub.columns), [list(map(str, r)) for r in sub.to_numpy()])
    for fig in sorted(FIGS.glob("*_cvar.png"))[:3]:
        _add_image(doc, fig, f"CVaR curve: {fig.stem}", width_in=5.2)
    cvar_bar = FIGS / "cvar_reduction.png"
    if cvar_bar.exists():
        _add_image(doc, cvar_bar,
                   "Figure: CVaR95 percent reduction vs. no-hedge, by fold and model",
                   width_in=6.0)

    # ============ 9a. Feature importance ============
    add_heading(doc, "9. Feature Importance", level=1)
    fi = _safe_read(TABLES / "feature_importance_top20.csv")
    if fi is not None and not fi.empty:
        for node in ["COMED", "PECO"]:
            sub = fi[fi["node"] == node].head(10)
            if sub.empty:
                continue
            add_heading(doc, f"9.1 Top-10 features — {node}", level=2)
            rows = [[str(r["feature"]), f"{float(r['gain_norm']):.4f}"]
                    for _, r in sub.iterrows()]
            add_table(doc, ["Feature", "Gain (normalized)"], rows)
            fig_fp = FIGS / f"feature_importance_{node.lower()}.png"
            if fig_fp.exists():
                _add_image(doc, fig_fp, f"Figure: top-20 XGBoost features, {node}", width_in=5.2)
    add_p(doc,
          "Feature gains are extracted from an XGBoost refit on the last fold's training "
          "window (2021-2023). Price lags and load magnitudes dominate; weather anomalies "
          "and storm-event counts add measurable marginal signal.")

    # ============ 10. Ablation ============
    add_heading(doc, "10. Ablation — Feature Contributions", level=1)
    ab = _safe_read(TABLES / "ablation_results.csv")
    if ab is not None and not ab.empty:
        xgb = ab[ab["model"] == "xgb_spike"].copy()
        keep = [c for c in ["node","fold_id","ablation","aucpr","aucroc","brier","lift_over_prevalence"] if c in xgb.columns]
        xgbfmt = xgb[keep].copy()
        for col in ["aucpr","aucroc","brier","lift_over_prevalence"]:
            if col in xgbfmt.columns:
                xgbfmt[col] = xgbfmt[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        add_table(doc, list(xgbfmt.columns), [list(map(str, r)) for r in xgbfmt.to_numpy()])

    # ============ 11. Discussion ============
    add_heading(doc, "11. Discussion and Limitations", level=1)
    add_bullets(doc, [
        "Weather-proxy leakage — using realized ERA5 observations rather than "
        "issued-at-d-1 forecasts biases AUCPR upward. This is the single largest source "
        "of optimism in our results. A drop-in swap to Meteomatics issued forecasts was "
        "designed in from the start (one config flag).",
        "Forward-price calibration for CVaR — our stylized $65/MWh DA-forward is too "
        "expensive for mild years (where realized DA averages $27/MWh), causing hedging "
        "to sometimes increase CVaR. The correct production fix is to use the realized "
        "DA-forward curve (price-of-risk) rather than a single placeholder.",
        "Quantile crossing — XGBoost's multi-alpha quantile head can produce P99 < P95; "
        "corrected post-hoc via monotone broadcast. A cleaner fix uses a composite "
        "quantile-loss objective that penalizes crossings during training.",
        "Storm-event feed is retrospective — in production an NWS real-time watch/warning "
        "feed would supplant the NCEI archive with full fidelity at forecast origin.",
        "Rare-label small-sample variance — absolute spike rates are ≤1% in mild years; "
        "AUCPR lifts are sharper but subject to CI in stress years (2021 February, 2022 "
        "summer).",
    ])

    # ============ 12. Conclusions ============
    add_heading(doc, "12. Conclusions", level=1)
    add_p(doc,
          "Day-ahead electricity price spikes in PJM can be predicted meaningfully above "
          "prevalence using a forecast-origin-clean feature panel that combines market lags, "
          "load forecasts, NOAA storm events, and weather. XGBoost with the full feature set "
          "is the strongest spike classifier in our rolling-origin evaluation; LEAR and MLP "
          "trail but provide valuable benchmarks. The upper-quantile XGBoost regressor is "
          "well-calibrated for P99 in most folds. Translated into the LSE's CVaR95 "
          "decision metric, a probability-triggered hedge rule materially reduces tail cost "
          "in stress years and is mildly counterproductive in mild years with a "
          "fixed-strike forward — a finding that motivates dynamic forward calibration as the "
          "next production step.")

    # ============ 13. Reproducibility ============
    add_heading(doc, "13. Reproducibility", level=1)
    add_p(doc,
          "Codebase: ~3,500 LOC across 35+ Python modules, 16 pytest unit tests. "
          "Entry points: scripts/run_mvp.sh runs the full 6-year, 3-fold, 2-node pipeline; "
          "scripts/run_smoke.sh runs a 3-month subset for CI-style smoke testing. "
          "Data sources: PJM DataMiner 2 API (free, user-specific subscription key), "
          "Open-Meteo historical archive (free, no auth), NOAA Storm Events bulk CSV "
          "(free, no auth). All raw data is cached as year-sharded parquet files; panel "
          "construction is deterministic given the same cache.")

    add_heading(doc, "13.1 Team contributions", level=2)
    add_table(doc, ["Member", "Primary contribution"], [
        ["Gavin Zeng", "Project framing, proposal drafting, event study, discussion"],
        ["Alex Yu",    "PECO secondary-node analysis, ablation, presentation lead"],
        ["Jialin Xu",  "Data pipeline (PJM, weather, NOAA), feature engineering, CVaR"],
        ["Andrew Lai", "Modeling (XGBoost, LEAR, MLP, LSTM), rolling-origin evaluation"],
    ])

    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
