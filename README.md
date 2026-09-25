# pjm-spike-forecasting

Forecast day-ahead electricity price spikes in PJM from weather and market data, then test whether those forecasts cut a power buyer's worst-case costs.

A utility that buys power for its customers pays the day-ahead price every hour. Most hours are cheap, and a few are spikes (here, an hour above $300/MWh or in the top 5% for its season). If the utility knows the day before that a spike is likely, it can lock in power at a fixed forward price instead.

This project builds that forecast and measures what it's worth.

## Results

On held-out years, a hedge triggered by the spike forecast cut the average cost of the worst 5% of days (CVaR95) by:

| Node | 2022 (stress year) | 2023 (mild year) | 2024 |
|---|---|---|---|
| COMED | **48.3%** | 11.3% | 21.1% |
| PECO | **57.7%** | 9.2% | 27.8% |

<img src="docs/figures/cvar_reduction.png" width="720">

- **Where it helps most:** stress years. In mild 2023, prices averaged $27/MWh against a $65 forward strike, so hedging rarely paid off.
- **Models:** no single model won everywhere. Logistic regression, XGBoost, LEAR (the standard electricity-price benchmark) and an MLP each won some node-year. In the stress year, logistic regression gave the largest cost reduction at both nodes.
- **Weather:** adding weather and storm features improved spike detection (AUCPR) over price-and-load-only models, most of all in the mixed-regime year.

## How it's built

- **No look-ahead.** Every feature is cut at 10:00 AM Eastern the day before, when a real buyer has to decide. A test corrupts future data and checks that predictions don't change.
- **Rolling evaluation.** Train on 2019–21 and test on 2022, then roll forward to 2023 and 2024. Spike thresholds are fit on training years only.
- **Data:** PJM Data Miner 2 (day-ahead prices, metered load, load forecasts), NOAA Storm Events and Open-Meteo weather. The hourly panel covers 2019–2024, with 52,608 rows per node.
- **Honest limits:** weather uses realized observations rather than the forecasts available at decision time, which flatters the results, and the fixed $65 forward strike simplifies a real forward curve. More under Known MVP limitations below.

Team project for UC Berkeley's Energy Analytics course (IND ENG 290), Spring 2026, with Alex Yu, Jialin Xu and Andrew Lai.

## 1. Setup

```bash
cd pjm-spike-forecasting

# Option A: dedicated conda env (recommended for clean Phase 2 builds)
conda env create -f environment.yml
conda activate energy
pip install -e .

# Option B: reuse the existing anaconda `base` env (quick)
/opt/anaconda3/bin/pip install -e .

# Provide PJM API key
cp .env.example .env
# Edit .env and set PJM_API_KEY=<your-primary-key from apiportal.pjm.com>
```

## 2. Smoke test (3-month subset)

```bash
./scripts/run_smoke.sh
```

### Using manually-downloaded DataMiner2 CSVs

Download one CSV per year per feed from https://dataminer2.pjm.com/ and drop
them in `data/raw/pjm_csv/` with these exact filenames:

```
da_hrl_lmps_comed_2019.csv         # Day-Ahead Hourly LMPs, Type=ZONE, Pnode=COMED
...                                # one per year
hrl_load_metered_comed_2019.csv    # Hourly Load Metered, Load Area=COMED
...
load_frcstd_hist_comed_2019.csv    # optional
```

Then convert them to the parquet cache the pipeline reads:

```bash
python -m ep_spikes.data.pjm_csv_ingest --zone COMED
./scripts/run_mvp.sh
```

### Without PJM data

Weather (Open-Meteo) and NOAA Storm Events work without auth. To exercise the
rest of the pipeline on **synthetic PJM data** for smoke-testing, run:

```bash
python -m ep_spikes.pipeline.step_01_fetch --config config/smoke.yaml --skip-pjm
python scripts/make_synthetic_pjm.py --years 2023 --zone COMED
./scripts/run_smoke.sh     # will pick up the synthetic cache
```

Outputs:
- `data/raw/{pjm,weather,noaa}/` — cached per-year parquet/CSV
- `data/processed/panel_comed.parquet` — feature panel
- `outputs/tables/results_mvp.csv` — model metrics
- `outputs/tables/cvar_summary.csv` — CVaR95 comparison
- `outputs/figures/*.png` — CVaR curves
- `outputs/reports/mvp_results.md` — rendered report

## 3. Full run

```bash
./scripts/run_mvp.sh
# Expect ~1–2 h wall clock; data fetches cache so subsequent runs are faster.
```

## 4. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefix avoids a click/flask version
conflict in the global anaconda env when pytest autoloads Dash.

## 5.

| Proposal spec | Implementation |
|---|---|
| PJM **day-ahead** hourly market | `src/ep_spikes/data/pjm_fetch.py` (`da_hrl_lmps` feed) |
| Primary node **COMED** | `config/mvp.yaml: data.nodes: [COMED]` |
| Spike label = **(price > $300)** OR **seasonal top-5%** | `src/ep_spikes/labels/spike.py::make_labels` — `label_abs`, `label_rel`, `label_either` |
| Seasonal threshold fit on **train only** | `compute_seasonal_thresholds` called inside the rolling loop (`eval/rolling.py:run_spike_fold`) |
| **Forecast origin** = 10 AM EPT day d-1 | `src/ep_spikes/tzutil.py::forecast_origin_utc` — DST-aware |
| Info-set respects origin | Every feature builder receives `origin` and is covered by `tests/test_features_leakage.py` |
| **Rolling-origin eval** 3 folds (2022/23/24) | `config/mvp.yaml: folds:` + `eval/rolling.py` |
| **Cross-entropy** for spike; **pinball** for quantile | `sklearn.LogisticRegression`, `XGBClassifier`, and `reg:quantileerror` |
| **AUCPR / Brier / reliability** | `eval/metrics.py`, `eval/plots.py` |
| Business metric: **CVaR95 of daily cost = Σ(L·P)** | `risk/cvar.py::simulate_hedge` |

## 6. Known limitations (flagged in `outputs/reports/mvp_results.md`)

- **Weather proxy.** Open-Meteo archive returns realized observations rather
  than issued-at-d-1 forecasts. This biases AUCPR upward. Phase 2 substitutes
  Meteomatics issued forecasts via a single config flip
  (`features.weather_forecast_proxy: meteomatics_issued`).
- **Quantile crossing.** XGBoost's `reg:quantileerror` can produce P99 < P95;
  we post-hoc enforce monotonicity per row. Phase 2 uses proper composite
  quantile training.
- **No deep learning.** LSTM/TCN deferred to Phase 2. The model interface
  (`fit` / `predict_proba` / `predict_quantiles`) is uniform so Phase 2 drops
  in without touching the rolling loop.
- **COMED only.** PECO added in Phase 2 for the congestion-vs-system-wide
  comparison described in the Clarification doc.

## 7. Repository layout

```
src/ep_spikes/         # importable package (installed editable)
  config.py            # YAML -> frozen dataclasses
  paths.py             # project directory constants
  tzutil.py            # EPT/DST, forecast_origin_utc, target_window_utc
  data/                # Open-Meteo, NOAA, PJM fetchers (per-year cache)
  features/            # calendar, prices, load, weather, storm, assemble
  labels/              # seasonal threshold + spike labels
  models/              # logistic, xgb_spike, xgb_quantile
  eval/                # rolling-origin loop, metrics, plots
  risk/                # CVaR + hedge-rule simulation
  pipeline/            # step_01..06 CLIs
config/                # mvp.yaml, smoke.yaml
scripts/               # run_mvp.sh, run_smoke.sh
tests/                 # pytest suite
notebooks/             # placeholders for EDA + diagnostics
outputs/               # models, predictions, figures, tables, reports
```

## 8. Quick reference: run individual steps

```bash
python -m ep_spikes.pipeline.step_01_fetch      --config config/smoke.yaml
python -m ep_spikes.pipeline.step_02_assemble   --config config/smoke.yaml
python -m ep_spikes.pipeline.step_03_label      --config config/smoke.yaml
python -m ep_spikes.pipeline.step_04_train_eval --config config/smoke.yaml
python -m ep_spikes.pipeline.step_05_cvar       --config config/smoke.yaml
python -m ep_spikes.pipeline.step_06_report     --config config/smoke.yaml

# Use --skip-pjm on step 1 to build only weather+NOAA while waiting for API key
python -m ep_spikes.pipeline.step_01_fetch --config config/smoke.yaml --skip-pjm
```
