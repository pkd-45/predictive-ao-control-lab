from __future__ import annotations

import json
from pathlib import Path


def load(tag: str):
    return json.loads((Path("results") / f"{tag}.json").read_text())["metrics"]


def idx(rows):
    return {r["controller"]: r for r in rows}


b = idx(load("baseline"))
s = idx(load("shifted"))
lines = [
    "# Validation result - 2026-08-23",
    "",
    (
        "This validation was run locally against the independent synthetic AO control demonstrator. "
        "The shifted regime reuses the **baseline-trained** predictor artifacts; "
        "it is not retrained on the shifted data."
    ),
    "",
    "## Nominal regime",
    "",
    "| Controller | Residual RMS | Improvement vs open loop | p95 latency (ms) | Fallbacks | Stable |",
    "|---|---:|---:|---:|---:|:---:|",
]
for name in ["open_loop", "integrator", "linear_predictor", "ml_safe"]:
    m = b[name]
    imp = 100 * (1 - m["residual_rms"] / b["open_loop"]["residual_rms"])
    lines.append(f"| {name} | {m['residual_rms']:.4f} | {imp:.1f}% | {m['latency_p95_ms']:.4f} | {m['fallback_count']} | {'yes' if m['stable'] else 'no'} |")
lines += ["", "## Distribution-shift regime (baseline models frozen)", "", "| Controller | Residual RMS | Change from nominal | p95 latency (ms) | Fallbacks | Stable |", "|---|---:|---:|---:|---:|:---:|"]
for name in ["open_loop", "integrator", "linear_predictor", "ml_safe"]:
    m = s[name]
    change = 100 * (m["residual_rms"] / b[name]["residual_rms"] - 1)
    lines.append(f"| {name} | {m['residual_rms']:.4f} | {change:+.1f}% | {m['latency_p95_ms']:.4f} | {m['fallback_count']} | {'yes' if m['stable'] else 'no'} |")
lines += [
    "",
    "## Interpretation",
    "",
    "- The classical integrator provides a strong closed-loop baseline against open loop.",
    "- The linear predictor is an important non-neural predictive baseline; the ML model is not assumed to win automatically.",
    "- The ML path is measured for software latency and runs behind a safety wrapper with deterministic fallback.",
    "- The changed atmospheric/noise regime probes robustness without allowing the predictive models to retrain on the test shift.",
    "- These results demonstrate software/control validation on a simplified synthetic plant, not METIS or ELT performance.",
]
Path("docs/VALIDATION_RESULT_2026-08-23.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
