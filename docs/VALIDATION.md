# Validation plan

The demonstrator is validated against software and control-oriented criteria rather than claimed instrument fidelity.

1. Reproducible atmospheric generation for fixed seeds.
2. Disjoint train/validation/test sequence identities.
3. Classical integrator improves residual RMS over open loop.
4. ML output shape and finite-value checks.
5. Safety wrapper falls back on NaN and actuator-limit violations.
6. SHA-256 artifact provenance changes when content changes.
7. Nominal and shifted test regimes are benchmarked with identical controller interfaces.
8. Report residual RMS, command RMS, saturation fraction, p50/p95/p99 call latency, fallback count, and a clearly labelled simplified Strehl proxy.

A lower training loss is not treated as sufficient evidence of a better controller; closed-loop residuals and stability are the primary acceptance measures.
