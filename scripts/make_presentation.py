"""Generate the class presentation .pptx from artifacts in outputs/.

Target: ~18 slides for a 15-minute classroom presentation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path("/Users/xujialin/Desktop/energy/Predicting_Electricity_Price_Spikes")
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
OUT = Path("/Users/xujialin/Desktop/IND_ENG_290_Presentation.pptx")

BERKELEY_BLUE = RGBColor(0x00, 0x32, 0x62)
BERKELEY_GOLD = RGBColor(0xFD, 0xB5, 0x15)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)


def _safe_read(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def _slide_title(slide, text: str, *, color=BERKELEY_BLUE, size: int = 32) -> None:
    title = slide.shapes.title
    if title is None:
        return
    title.text = text
    for para in title.text_frame.paragraphs:
        para.alignment = PP_ALIGN.LEFT
        for run in para.runs:
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = color


def _bullets(slide, items: list[str], *, size: int = 18, left: float = 0.5,
             top: float = 1.5, width: float = 9.0, height: float = 5.0) -> None:
    tx = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tx.text_frame; tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.text = t
        for run in p.runs:
            run.font.size = Pt(size); run.font.color.rgb = DARK_GRAY


def _image(slide, img: Path, left: float, top: float, width: float) -> None:
    if img.exists():
        slide.shapes.add_picture(str(img), Inches(left), Inches(top), width=Inches(width))


def _notes(slide, text: str) -> None:
    """Attach speaker notes to the slide."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text


def _table(slide, df: pd.DataFrame, left: float, top: float, width: float, height: float,
           max_rows: int = 10) -> None:
    df = df.head(max_rows)
    rows, cols = df.shape[0] + 1, df.shape[1]
    tbl = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                 Inches(width), Inches(height)).table
    for j, col in enumerate(df.columns):
        c = tbl.cell(0, j)
        c.text = str(col)
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
        c.fill.solid(); c.fill.fore_color.rgb = BERKELEY_BLUE
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            c = tbl.cell(i + 1, j)
            c.text = "—" if pd.isna(v) else str(v)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11); r.font.color.rgb = DARK_GRAY


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_layout = prs.slide_layouts[0]
    content_layout = prs.slide_layouts[5]  # Title + blank content
    blank_layout = prs.slide_layouts[6]

    # -------- Slide 1: Title --------
    s = prs.slides.add_slide(title_layout)
    if s.shapes.title:
        s.shapes.title.text = "Predicting Electricity Price Spikes\nfrom Extreme Weather in PJM"
        for p in s.shapes.title.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = BERKELEY_BLUE
    if len(s.placeholders) > 1:
        sub = s.placeholders[1]
        sub.text = ("Deep Learning Forecasts and Cost-at-Risk Quantification\n"
                    "Gavin Zeng · Alex Yu · Jialin Xu · Andrew Lai\n"
                    "INDENG 290 — Energy Analytics · Spring 2026")
        for p in sub.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(18); r.font.color.rgb = DARK_GRAY

    # -------- 2: Motivation --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Why tail-price forecasting matters for an LSE")
    _bullets(s, [
        "• Day-ahead electricity prices in PJM average about $30/MWh — but extreme weather "
        "pushes them past $1,000/MWh for contiguous hours.",
        "• These tail events are rare (≈1% of hours) but drive a disproportionate share of "
        "annual procurement cost for a load-serving entity (LSE).",
        "• Standard forecasting models underpredict spikes; the LSE is left poorly hedged.",
        "• Question: can explicit weather + storm signals + modern ML improve tail-risk "
        "prediction, and does it cut Cost-at-Risk (CVaR95) for the client?",
    ])
    _notes(s, "Open with the concrete: normal PJM day-ahead LMPs sit in the $20–60 range, "
              "but when scarcity hits you can see hours over $1,000. That asymmetric tail is "
              "what makes electricity procurement risky — not mean error. Frame the problem "
              "as: we care about the top 1% of hours, and standard regression-style "
              "forecasters underestimate them systematically.")

    # -------- 3: Research questions --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Research questions")
    _bullets(s, [
        "1. How accurately can we predict next-day spike probability and upper quantiles "
        "(P95 / P99)?",
        "2. How much do explicit weather + storm features add over market-only?",
        "3. Do XGBoost, LEAR, MLP, LSTM outperform a logistic baseline?",
        "4. How much CVaR95 reduction does the improved prediction buy the LSE?",
    ])

    # -------- 4: Client & CVaR definition --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Client: a stylized PJM LSE")
    _bullets(s, [
        "• Hourly procurement cost C_t = L_t × P_t",
        "• Daily cost C_d = Σ_h L_h × P_h over the 24 (or 23/25 on DST) hours of day d",
        "• Primary decision metric: CVaR95 of the distribution of C_d over the test year",
        "• Hedge rule: if predicted spike probability ≥ threshold t, lock in DA-forward "
        "price (stylized $65/MWh); otherwise pay the realized DA LMP",
        "• Improved prediction → better hedge timing → lower tail cost",
    ])

    # -------- 5: Data --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Data")
    _bullets(s, [
        "• PJM DataMiner 2 — da_hrl_lmps  (DA LMPs, COMED + PECO)",
        "• PJM DataMiner 2 — hrl_load_metered  (COMED metered load, 'CE' subzone alias)",
        "• PJM DataMiner 2 — load_frcstd_hist  (as-of DA load forecast)",
        "• Open-Meteo ERA5 archive  (hourly weather, Chicago + Philadelphia)",
        "• NOAA Storm Events DB  (severe weather indicators, 14 PJM-footprint states)",
        "• Window: 2019-01-01 through 2024-12-31 (six calendar years)",
    ])
    _notes(s, "All data pulled programmatically. PJM via DataMiner 2 REST; Open-Meteo is "
              "free and no-auth; NOAA storm events via their bulk CSV FTP. COMED load uses "
              "the CE subzone code (Commonwealth Edison); PECO corresponds to PE. Our "
              "pipeline caches per-year parquet to avoid re-hitting the API on re-runs.")

    # -------- 6: Forecast-origin information set --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Forecast origin: 10:00 AM EPT on day d − 1")
    _bullets(s, [
        "• Aligns with the PJM DA bidding deadline at noon EPT.",
        "• At origin we know: all settled DA LMPs through d − 1, PJM published DA load "
        "forecast for day d, day-ahead weather forecast for day d, NOAA storm events with "
        "begin_utc < origin.",
        "• We do NOT know: any target-hour realized values for day d.",
        "• Enforced programmatically: every feature builder takes an explicit origin "
        "timestamp; pytest corrupts post-origin data and asserts features don't change.",
    ])
    _notes(s, "This slide is the technical integrity story. Many spike-forecast papers leak "
              "future information through rolling statistics or label thresholds; we close "
              "both holes with explicit origin arguments and a unit test that *perturbs "
              "post-origin data* and checks that features don't move. DST also handled (23h "
              "spring-forward, 25h fall-back days) — non-trivial for PJM.")

    # -------- 7: Feature engineering --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Feature engineering")
    _bullets(s, [
        "Calendar — hour, dow, month, season, holiday, cyclic sin/cos",
        "Prices — lags 24/48/168 h; rolling mean/std/max/min over 24 and 168 h; 24-h ramp",
        "Load — DA forecast (as-of issue ≤ origin) + lagged metered (to d-2) + forecast-to-last ratio",
        "Weather — hourly temp / humidity / precip / wind / gusts + daily aggregates + temp anomaly Z",
        "Storms — NOAA event counts (begin_utc < origin) in 12/24/48 h, plus type-specific flags",
    ])

    # -------- 8: Labels --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Spike labels")
    _bullets(s, [
        "label_abs  —  LMP > $300/MWh (absolute scarcity threshold)",
        "label_rel  —  top 5% within season (summer, winter, shoulder)",
        "label_either  —  OR of the two; primary training target",
        "• Seasonal threshold is fit on each fold's TRAIN window only — never on the full "
        "panel. Unit test asserts this.",
    ])

    # -------- 9: Models --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Models")
    _bullets(s, [
        "Logistic regression — baseline, median-impute → standardize → LR",
        "XGBoost classifier — small grid, scale_pos_weight auto, AUCPR selection",
        "XGBoost quantile — reg:quantileerror, alphas [0.95, 0.99], monotone fix",
        "LEAR (Lasso-AR + isotonic calibrator) — transparent econometric benchmark",
        "MLP (fully connected, 128-64) — \"deep learning lite\"",
        "LSTM (torch, 24-hour sliding window) — sequence model, optional install",
    ])
    _notes(s, "All five classifiers expose the same .fit / .predict_proba interface so the "
              "rolling-origin loop swaps them freely. LEAR is our econometric anchor — "
              "transparent, close to the EPFtoolbox reference. MLP is the \"deep learning "
              "lite\" version of LSTM/TCN; since our features already encode temporal "
              "context via lags, a fully-connected net recovers most of the sequence signal.")

    # -------- 10: Rolling-origin eval --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Rolling-origin evaluation protocol")
    _bullets(s, [
        "Fold 1:  train 2019-01 → 2021-12  →  test 2022",
        "Fold 2:  train 2020-01 → 2022-12  →  test 2023",
        "Fold 3:  train 2021-01 → 2023-12  →  test 2024",
        "• Fixed 3-year window prevents distributional drift from far history",
        "• Hyperparameter tuning via TimeSeriesSplit(4) inside train window only",
        "• No test data ever seen by tuner or label-threshold fitter",
    ])

    # -------- 11: Event-study results --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Event study: storm features matter")
    ev = _safe_read(TABLES / "event_study_comed.csv")
    if ev is not None and not ev.empty:
        sig = ev[ev["is_event"] == True].sort_values("p").head(7).copy()
        for c in ["coef","std_err","t","p"]:
            if c in sig.columns:
                sig[c] = sig[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
        _table(s, sig[["variable","coef","std_err","t","p"]],
               left=0.5, top=1.5, width=12.3, height=4.5, max_rows=8)
    _bullets(s, [
        "OLS on training fold; log(1+LMP) ~ event indicators + hour × dow × month FE, HC1 SE",
    ], top=6.3, size=14)

    # -------- 12: Spike classification results --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Spike-probability classification — held-out AUCPR")
    res = _safe_read(TABLES / "results_mvp.csv")
    if res is not None and not res.empty:
        sp = res[res["task"] == "spike"].copy()
        keep = [c for c in ["node","fold_id","model","prevalence","aucpr","aucroc","lift_over_prevalence"] if c in sp.columns]
        sp = sp[keep]
        for c in ["prevalence","aucpr","aucroc","lift_over_prevalence"]:
            if c in sp.columns:
                sp[c] = sp[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        _table(s, sp, left=0.5, top=1.5, width=12.3, height=4.5, max_rows=14)
    _bullets(s, [
        "Lift over prevalence (AUCPR / prevalence) measures precision-recall lift over the "
        "\"always predict prevalence\" benchmark.",
    ], top=6.3, size=14)

    # -------- 13: Reliability diagrams --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Calibration — reliability diagrams")
    pngs = sorted(FIGS.glob("*reliability*.png"))[:2]
    if pngs:
        _image(s, pngs[0], left=0.5, top=1.5, width=6.0)
    if len(pngs) > 1:
        _image(s, pngs[1], left=7.0, top=1.5, width=6.0)
    _bullets(s, [
        "Empirical event rate vs. predicted probability, 10 equal-width bins",
        "Closer to the 45° diagonal = better-calibrated probabilities",
    ], top=6.1, size=14)

    # -------- 14: Quantile forecast --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Upper-quantile forecast coverage (P95, P99)")
    if res is not None:
        q = res[res["task"] == "quantile"].copy()
        keep = [c for c in ["node","fold_id","coverage_q95","coverage_q99","pinball_q95","pinball_q99"] if c in q.columns]
        q = q[keep]
        for c in ["coverage_q95","coverage_q99","pinball_q95","pinball_q99"]:
            if c in q.columns:
                q[c] = q[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        _table(s, q, left=0.5, top=1.5, width=12.3, height=3.5, max_rows=8)
    _bullets(s, [
        "Coverage @α ≈ α is the calibration target (P95 ≈ 0.95, P99 ≈ 0.99)",
        "Pinball loss at α = mean of α·u⁺ + (1-α)·u⁻ ; lower is better",
    ], top=5.8, size=14)

    # -------- 15: CVaR business metric --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Business metric: CVaR95 of daily procurement cost")
    cv = _safe_read(TABLES / "cvar_summary.csv")
    if cv is not None and not cv.empty:
        keep = [c for c in ["node","fold_id","model_tag","strategy","threshold","cvar","mean_daily_cost","n_triggered_hours"] if c in cv.columns]
        cv_fmt = cv[keep].copy()
        for c in ["cvar","mean_daily_cost"]:
            if c in cv_fmt.columns:
                cv_fmt[c] = cv_fmt[c].map(lambda v: f"${v/1e6:.2f}M" if pd.notna(v) else "—")
        if "threshold" in cv_fmt.columns:
            cv_fmt["threshold"] = cv_fmt["threshold"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        _table(s, cv_fmt, left=0.2, top=1.5, width=13.0, height=4.5, max_rows=12)

    # -------- 16: CVaR curve figure --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "CVaR95 vs. hedge threshold")
    pngs = sorted(FIGS.glob("*cvar.png"))[:2]
    if pngs:
        _image(s, pngs[0], left=0.5, top=1.5, width=6.0)
    if len(pngs) > 1:
        _image(s, pngs[1], left=7.0, top=1.5, width=6.0)
    _bullets(s, [
        "Each point: CVaR95 over the test-year daily-cost distribution when hedging at "
        "prob ≥ threshold with a $65/MWh stylized forward",
        "Dashed line: no-hedge baseline",
    ], top=6.1, size=14)

    # -------- 16b: Key finding --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Key finding")
    _bullets(s, [
        "• In stress-year folds (2022 heat wave, 2024 winter events), a XGBoost-triggered "
        "hedge materially lowers CVaR95 vs. the unhedged baseline.",
        "• In mild-year folds (2023), the stylized $65/MWh forward is too expensive and "
        "hedging slightly increases CVaR95 — a calibration issue, not a model issue.",
        "• Fix is immediate: replace the fixed forward with the realized DA-forward curve "
        "from futures markets — already a Phase 2 design seam.",
    ])
    _notes(s, "This is the nuanced business takeaway. The spike classifier is good; the "
              "hedge math needs a data-driven forward curve. Make sure to tell the audience "
              "this is a *finding*, not a failure — it teaches something about operational "
              "risk that a pure accuracy metric would hide.")

    # -------- 17: Ablation --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Ablation — what moves AUCPR?")
    ab = _safe_read(TABLES / "ablation_results.csv")
    if ab is not None and not ab.empty:
        xg = ab[ab["model"] == "xgb_spike"].copy()
        keep = [c for c in ["node","fold_id","ablation","aucpr","aucroc","lift_over_prevalence"] if c in xg.columns]
        xg = xg[keep]
        for c in ["aucpr","aucroc","lift_over_prevalence"]:
            if c in xg.columns:
                xg[c] = xg[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        _table(s, xg, left=0.5, top=1.5, width=12.3, height=4.0, max_rows=12)
    _bullets(s, [
        "market_only = price + load lags   |   +weather = add Open-Meteo   |   +storms = add NOAA",
    ], top=5.9, size=14)

    # -------- 18: Limitations & conclusions --------
    s = prs.slides.add_slide(content_layout)
    _slide_title(s, "Limitations and conclusions")
    _bullets(s, [
        "Weather proxy biases AUCPR up — Phase 2 drop-in: Meteomatics issued forecasts",
        "Fixed $65 forward overpays in mild years — dynamic DA-forward curve would fix CVaR sign",
        "Quantile crossing handled post-hoc; composite quantile loss is cleaner",
        "Main takeaway: ML tail-risk forecasts translate into material CVaR95 reduction in "
        "stress years — validates the LSE use case for data-driven hedging",
        "Code + data pipeline is fully reproducible; Phase 2 seams already in place for LSTM, "
        "LEAR, ablation, and event study",
    ])
    _notes(s, "Close with honesty about what we would improve next and re-state the business "
              "punchline: on stress years the hedge materially lowers CVaR, which is exactly "
              "what the LSE client asked for. The limitations are concrete and actionable, "
              "not hand-wavy.")

    # -------- 19: Thank you --------
    s = prs.slides.add_slide(blank_layout)
    tx = s.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(11), Inches(3))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.text = "Thank you"
    for r in p.runs:
        r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = BERKELEY_BLUE
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    p2.text = "Questions?"
    for r in p2.runs:
        r.font.size = Pt(28); r.font.color.rgb = DARK_GRAY

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
