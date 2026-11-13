# Predictive AO Control Lab

A compact, independent synthetic adaptive-optics control demonstrator built to explore how a machine-learning predictor can be integrated into an RTC-like software boundary with classical baselines, safety fallbacks, latency measurements, provenance, automated tests, and reproducible benchmark records.

> **Scope boundary:** this is not METIS/ELT production software and not a physical model of the METIS real-time controller. It uses a deliberately simplified modal disturbance and control loop to demonstrate software and validation concepts.

## Why this exists

The project targets five transferable engineering questions:

1. Can an ML predictor be compared fairly with classical and linear predictive baselines?
2. Can offline training be cleanly separated from online inference?
3. Can the inference path fail safely to a deterministic controller?
4. Can closed-loop quality and software latency be measured together?
5. Can experiment/model provenance be made machine-readable and reproducible?

## Architecture

```text
synthetic atmosphere -> noisy WFS measurement -> controller -> bounded DM command
                                            |                  |
                                            +-> telemetry/QA <-+

controllers: open loop | integrator | linear predictor | ML predictor + safe fallback
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

./scripts/run_validation.sh

# Equivalent explicit commands:
python -m ao_lab all --config configs/baseline.yaml --tag baseline
python -m ao_lab generate --config configs/distribution_shift.yaml --tag shifted
python -m ao_lab evaluate --config configs/distribution_shift.yaml --tag shifted --model-tag baseline
pytest
```

Outputs are written to `results/`, with model artifacts in `models/`, datasets in `data/`, and a benchmark database at `results/benchmarks.sqlite3`.

## Metrics

- residual modal RMS;
- simplified Strehl proxy `exp(-residual_rms^2)`;
- command RMS;
- actuator saturation fraction;
- controller-call latency p50/p95/p99;
- safety-fallback count;
- stability flag.

## Validation summary

The benchmark compares an ML predictive controller with classical integrator
and ridge-predictive baselines in a deliberately simplified closed-loop
simulation. Offline training is separate from online inference, and the
ML path has a deterministic fallback for invalid or out-of-range outputs.

Reference experiments include nominal and shifted disturbance regimes,
numerical quality checks, latency measurements and recorded model provenance.
The linear predictor outperformed the MLP in the measured nominal reference
configuration; ML is not assumed to be the preferred controller.

See [technical capabilities](docs/CAPABILITIES.md) and
[validation evidence](docs/VALIDATION_RESULT_2026-08-23.md).
