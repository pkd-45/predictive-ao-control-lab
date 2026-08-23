#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m compileall -q src
pytest -q

# Nominal: train and evaluate on disjoint baseline train/val/test sequences.
python -m ao_lab all --config configs/baseline.yaml --tag baseline

# Distribution shift: generate a changed atmosphere/noise regime, but reuse the
# baseline-trained linear and ML artifacts. This tests robustness rather than retraining.
python -m ao_lab generate --config configs/distribution_shift.yaml --tag shifted
python -m ao_lab evaluate --config configs/distribution_shift.yaml --tag shifted --model-tag baseline

python scripts/summarize_validation.py
