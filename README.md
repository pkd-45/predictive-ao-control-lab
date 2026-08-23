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

## Application-safe description

> In response to an adaptive-optics/RTC software role, I built an independent synthetic control demonstrator that separates offline ML training from an RTC-like inference path and compares a PyTorch predictive controller with classical and linear baselines. The software records closed-loop residual error and inference latency, applies numerical/actuator/latency safety checks, and falls back to a deterministic controller when ML output is invalid. I benchmarked both nominal and distribution-shift regimes and packaged the workflow with configuration-driven data generation, automated tests, SHA-256 model provenance and a SQLite benchmark record. It is deliberately a simplified software/control demonstrator rather than a model of the METIS RTC.
