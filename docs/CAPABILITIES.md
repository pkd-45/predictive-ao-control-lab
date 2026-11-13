# Predictive AO control: implemented capabilities

This repository implements a **synthetic** modal-control benchmark. It is not
METIS/ELT production software, a physical METIS real-time controller, or
evidence of observatory commissioning.

## Model development and data flow

- PyTorch MLP predictor with deterministic sequence generation and disjoint
  train/validation/test identities
- Configuration-driven HDF5 data and command-line workflow
- Separate offline training and online inference behind a safe controller
- Classical integrator and ridge-regression predictive baselines

## Runtime and validation

- Closed-loop residual RMS, command RMS and actuator saturation
- p50/p95/p99 Python controller-call latency and stability checks
- Deterministic fallback after NaNs and actuator-limit failures
- Distribution-shift benchmark using frozen baseline-trained models
- SQLite benchmark records, model hashes, CI, Ruff, MyPy and pytest

## Recorded reference results (23 August 2026)

- Seven of seven tests passed in the documented run.
- Nominal residual RMS: open loop 0.9426; integrator 0.3536;
  linear predictor 0.3053; safe MLP 0.3266.
- Nominal safe-ML Python controller-call p95 latency 0.0373 ms on the
  documented Mac system.
- The linear model outperformed the MLP in the nominal toy plant.
  This result motivates baseline-first evaluation, not an ML superiority claim.

See `VALIDATION_RESULT_2026-08-23.md` for the execution record.
Latency is machine-specific Python timing, **not** guaranteed hard-real-time
METIS performance. The repository does not demonstrate pyramid-WFS modelling,
real deformable-mirror operation, commissioning or measured GPU acceleration.
