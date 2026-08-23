# Design

## Scope

This repository is an independent synthetic adaptive-optics software/control demonstrator. It is **not** a METIS, ELT, ESO, or MPIA simulator and contains no proprietary interfaces or data.

## Control model

The synthetic plant represents a small vector of modal wavefront coefficients with coupled AR(1) temporal evolution. The previous deformable-mirror command is subtracted from the current phase to produce the residual; a noisy synthetic wavefront-sensor measurement is passed to a controller. Commands affect the next step, creating a one-step-delay closed loop.

## Controllers

- open-loop reference;
- classical integrator;
- ridge-regression linear predictor;
- PyTorch MLP predictive controller behind `SafeController`.

The ML controller is trained offline. The online path loads a frozen artifact and performs inference only.

## Safety boundary

`SafeController` rejects non-finite predictions, commands outside the actuator bound, exceptions, and inference calls beyond a configured latency budget. Rejected predictions fall back to a deterministic integrator.

## Engineering outputs

The evaluation writes machine-readable JSON, SHA-256 model provenance, a SQLite benchmark database, and plots. Nominal and distribution-shift regimes use separate configuration files.
