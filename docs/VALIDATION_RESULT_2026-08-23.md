# Validation result - 2026-08-23

This validation was run locally against the independent synthetic AO control demonstrator. The shifted regime reuses the **baseline-trained** predictor artifacts; it is not retrained on the shifted data.

## Nominal regime

| Controller | Residual RMS | Improvement vs open loop | p95 latency (ms) | Fallbacks | Stable |
|---|---:|---:|---:|---:|:---:|
| open_loop | 0.9426 | 0.0% | 0.0003 | 0 | yes |
| integrator | 0.3536 | 62.5% | 0.0020 | 0 | yes |
| linear_predictor | 0.3053 | 67.6% | 0.0064 | 0 | yes |
| ml_safe | 0.3266 | 65.4% | 0.0373 | 0 | yes |

## Distribution-shift regime (baseline models frozen)

| Controller | Residual RMS | Change from nominal | p95 latency (ms) | Fallbacks | Stable |
|---|---:|---:|---:|---:|:---:|
| open_loop | 0.8603 | -8.7% | 0.0003 | 0 | yes |
| integrator | 0.4852 | +37.2% | 0.0020 | 0 | yes |
| linear_predictor | 0.4336 | +42.0% | 0.0065 | 0 | yes |
| ml_safe | 0.4464 | +36.7% | 0.0377 | 0 | yes |

## Interpretation

- The classical integrator provides a strong closed-loop baseline against open loop.
- The linear predictor is an important non-neural predictive baseline; the ML model is not assumed to win automatically.
- The ML path is measured for software latency and runs behind a safety wrapper with deterministic fallback.
- The changed atmospheric/noise regime probes robustness without allowing the predictive models to retrain on the test shift.
- These results demonstrate software/control validation on a simplified synthetic plant, not METIS or ELT performance.
