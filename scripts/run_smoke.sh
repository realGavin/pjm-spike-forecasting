#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Running smoke test (3-month window, COMED only)"

python -m ep_spikes.pipeline.step_01_fetch     --config config/smoke.yaml
python -m ep_spikes.pipeline.step_02_assemble  --config config/smoke.yaml
python -m ep_spikes.pipeline.step_03_label     --config config/smoke.yaml
python -m ep_spikes.pipeline.step_04_train_eval --config config/smoke.yaml
python -m ep_spikes.pipeline.step_05_cvar      --config config/smoke.yaml
python -m ep_spikes.pipeline.step_06_report    --config config/smoke.yaml

echo ""
echo "Smoke done. See outputs/reports/mvp_results.md"
