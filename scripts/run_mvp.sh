#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/mvp.yaml}"
cd "$(dirname "$0")/.."
echo "Running full MVP with config: $CONFIG"

python -m ep_spikes.pipeline.step_01_fetch     --config "$CONFIG"
python -m ep_spikes.pipeline.step_02_assemble  --config "$CONFIG"
python -m ep_spikes.pipeline.step_03_label     --config "$CONFIG"
python -m ep_spikes.pipeline.step_04_train_eval --config "$CONFIG"
python -m ep_spikes.pipeline.step_05_cvar      --config "$CONFIG"
python -m ep_spikes.pipeline.step_06_report    --config "$CONFIG"

echo ""
echo "Done. See outputs/reports/mvp_results.md"
