# MPIA METIS RTC application mapping

Target: **MPIA Software Developer / Computer Scientist specialised in AI & Machine Learning, Ref. 421**.

This file maps the independent demonstrator to the advertised tasks without claiming METIS/ESO production experience.

## Advertised task -> demonstrator evidence

| MPIA task/profile | Evidence in this repository |
|---|---|
| Model development: design, training and validation of ML/DL models | PyTorch MLP predictor; deterministic dataset generation; disjoint train/validation/test sequences; validation-selected artifact |
| Pipeline architecture / preparation of training data | Configuration-driven HDF5 dataset generation; explicit train/val/test identities; CLI workflow from generation to benchmark |
| AI integration into existing software architectures | Common controller interface; ML predictor integrated behind a `SafeController`; model artifact loaded only in the online path |
| Evaluation and optimisation | Closed-loop residual RMS, command RMS, actuator saturation, p50/p95/p99 controller-call latency, stability flag, comparison with open-loop/classical/linear baselines |
| Tech scouting / appropriate method choice | Classical integrator and ridge predictive controller retained as strong non-neural baselines; MLP is not assumed to be superior |
| Laboratory/system test mindset | Failure injection tests for NaN and actuator-limit violations; deterministic fallback; distribution-shift benchmark with frozen baseline-trained artifacts |
| Python / ML libraries | Python, NumPy, PyTorch, h5py, Matplotlib |
| SQL / data systems | SQLite benchmark database records controller/regime/quality/latency/model hash |
| Git / CI/CD | Git-ready repository; GitHub Actions matrix for Python 3.10-3.12 with Ruff, MyPy and pytest |
| International/commissioning readiness | This repository does not establish commissioning experience; application should rely on existing observatory/observing evidence for that criterion |

## Locally validated result

`./scripts/run_validation.sh` completed successfully on 2026-08-23:

- **7/7 pytest tests passed**.
- Nominal closed-loop residual RMS: open loop **0.9426**, integrator **0.3536**, linear predictor **0.3053**, safe MLP **0.3266**.
- Nominal MLP controller-call p95 latency: **0.0756 ms** on the local validation environment.
- Distribution-shift benchmark reused the baseline-trained predictive artifacts; all four controller paths remained numerically stable.
- In the deliberately simple mostly linear synthetic plant, the linear predictor outperformed the MLP. This is treated as an informative baseline result, not hidden or tuned away.

Latency values are environment-specific software-call timings and **must not** be described as METIS RTC latency or hard-real-time performance.

## Safe application wording

> To make my transfer into adaptive-optics control software inspectable, I built an independent synthetic predictive-control demonstrator. It separates offline PyTorch training from an RTC-like inference path, compares the learned predictor with classical-integrator and ridge-predictive baselines, records closed-loop residuals and controller-call latency, and places the ML path behind numerical, actuator and latency checks with deterministic fallback. I also tested a changed atmosphere/noise regime using the frozen baseline-trained artifacts. On this deliberately simple synthetic plant the linear predictor outperformed the neural model, which I regard as a useful result: I would not introduce ML complexity without evidence that it improves the control objective. The project is a software/control demonstrator, not a model of the METIS RTC.

## Claims not supported by this repository

Do not claim:

- METIS or ESO software experience;
- physical ELT/METIS AO fidelity;
- pyramid-wavefront-sensor modelling;
- real deformable-mirror hardware control;
- hard real-time scheduling guarantees;
- measured GPU acceleration;
- laboratory AO-loop closure;
- commissioning experience from this repository.
