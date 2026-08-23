from __future__ import annotations

import json
from pathlib import Path


def markdown_table(result_json: str | Path) -> str:
    payload = json.loads(Path(result_json).read_text())
    lines = [
        "| Controller | Residual RMS | Strehl proxy | p95 latency (ms) | Fallbacks | Stable |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for m in payload["metrics"]:
        lines.append(
            f"| {m['controller']} | {m['residual_rms']:.4f} | {m['strehl_proxy']:.4f} | "
            f"{m['latency_p95_ms']:.4f} | {m['fallback_count']} | {'yes' if m['stable'] else 'no'} |"
        )
    return "\n".join(lines)
