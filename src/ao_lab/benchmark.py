from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt


def write_sqlite(result_json: str | Path, db_path: str | Path, regime: str) -> Path:
    payload = json.loads(Path(result_json).read_text())
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
          regime TEXT NOT NULL,
          controller TEXT NOT NULL,
          residual_rms REAL NOT NULL,
          strehl_proxy REAL NOT NULL,
          command_rms REAL NOT NULL,
          saturation_fraction REAL NOT NULL,
          latency_p50_ms REAL NOT NULL,
          latency_p95_ms REAL NOT NULL,
          latency_p99_ms REAL NOT NULL,
          fallback_count INTEGER NOT NULL,
          stable INTEGER NOT NULL,
          model_sha256 TEXT,
          PRIMARY KEY (regime, controller, model_sha256)
        )
        """
    )
    model_sha = payload["artifacts"]["model_sha256"]
    for m in payload["metrics"]:
        con.execute(
            """INSERT OR REPLACE INTO benchmark_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                regime, m["controller"], m["residual_rms"], m["strehl_proxy"], m["command_rms"],
                m["saturation_fraction"], m["latency_p50_ms"], m["latency_p95_ms"],
                m["latency_p99_ms"], m["fallback_count"], int(m["stable"]), model_sha,
            ),
        )
    con.commit(); con.close()
    return db_path


def make_plots(result_json: str | Path, output_dir: str | Path) -> None:
    payload = json.loads(Path(result_json).read_text())
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    names = [m["controller"] for m in metrics]
    vals = [m["residual_rms"] for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals)
    ax.set_ylabel("Residual modal RMS")
    ax.set_title("Closed-loop controller comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "controller_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, trace in payload["traces"].items():
        ax.plot(trace["residual_rms_time"], label=name, linewidth=1)
    ax.set_xlabel("Time step"); ax.set_ylabel("Residual RMS")
    ax.set_title("Representative synthetic closed-loop episode")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "residual_timeseries.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, trace in payload["traces"].items():
        if name == "open_loop":
            continue
        ax.hist(trace["latency_ms"], bins=35, alpha=0.45, label=name)
    ax.set_xlabel("Controller call latency [ms]"); ax.set_ylabel("Count")
    ax.set_title("RTC-like software latency benchmark")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "latency_distribution.png", dpi=150); plt.close(fig)
