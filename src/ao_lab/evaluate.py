from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import Config
from .controllers import (
    Controller,
    IntegratorController,
    LinearPredictiveController,
    MLPredictiveController,
    OpenLoopController,
    SafeController,
)
from .dataset import load_split
from .train import load_mlp


@dataclass
class ControllerMetrics:
    controller: str
    residual_rms: float
    strehl_proxy: float
    command_rms: float
    saturation_fraction: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    fallback_count: int
    stable: bool


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_episode(cfg: Config, phase: np.ndarray, controller, seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    controller.reset()
    command = np.zeros(cfg.n_modes, dtype=np.float64)
    residuals: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    latencies: list[float] = []
    for x in phase:
        residual = x - command
        measurement = residual + rng.normal(0.0, cfg.wfs_noise_sigma, size=cfg.n_modes)
        start = perf_counter()
        next_command = controller.command(measurement, command)
        elapsed_ms = (perf_counter() - start) * 1e3
        latencies.append(elapsed_ms)
        residuals.append(residual.copy())
        commands.append(next_command.copy())
        command = next_command
    return np.asarray(residuals), np.asarray(commands), latencies


def summarize(cfg: Config, name: str, residuals: np.ndarray, commands: np.ndarray, latencies: list[float], fallback_count: int = 0) -> ControllerMetrics:
    # Skip a short warmup to avoid penalising history-based controllers before history exists.
    warm = max(cfg.history + 2, 8)
    r = residuals[:, warm:, :]
    c = commands[:, warm:, :]
    residual_rms = float(np.sqrt(np.mean(r**2)))
    command_rms = float(np.sqrt(np.mean(c**2)))
    saturation_fraction = float(np.mean(np.abs(c) >= 0.999 * cfg.actuator_limit))
    lat = np.asarray(latencies, dtype=float)
    return ControllerMetrics(
        controller=name,
        residual_rms=residual_rms,
        strehl_proxy=float(np.exp(-residual_rms**2)),
        command_rms=command_rms,
        saturation_fraction=saturation_fraction,
        latency_p50_ms=float(np.percentile(lat, 50)),
        latency_p95_ms=float(np.percentile(lat, 95)),
        latency_p99_ms=float(np.percentile(lat, 99)),
        fallback_count=int(fallback_count),
        stable=bool(np.isfinite(residual_rms) and residual_rms < 5.0 and saturation_fraction < 0.5),
    )


def evaluate_all(
    cfg: Config,
    dataset_path: str | Path,
    linear_path: str | Path,
    model_path: str | Path,
    output_json: str | Path,
) -> list[ControllerMetrics]:
    phases, seeds = load_split(dataset_path, "test")
    linear = np.load(linear_path)
    model = load_mlp(model_path)
    controllers: list[Controller] = [
        OpenLoopController(cfg.n_modes),
        IntegratorController(cfg.integrator_gain, cfg.actuator_limit),
        LinearPredictiveController(cfg.history, cfg.n_modes, linear["weights"], linear["bias"], cfg.actuator_limit),
        SafeController(
            MLPredictiveController(cfg.history, cfg.n_modes, model, cfg.actuator_limit),
            IntegratorController(cfg.integrator_gain, cfg.actuator_limit),
            cfg.actuator_limit,
            cfg.latency_budget_ms,
        ),
    ]
    rows: list[ControllerMetrics] = []
    traces: dict[str, dict[str, list]] = {}
    for controller in controllers:
        all_r: list[np.ndarray] = []
        all_c: list[np.ndarray] = []
        all_l: list[float] = []
        fallback_total = 0
        for phase, seed in zip(phases, seeds):
            r, c, l = run_episode(cfg, phase, controller, int(seed) + 777)
            all_r.append(r); all_c.append(c); all_l.extend(l)
            fallback_total += int(getattr(controller, "fallback_count", 0))
        r_arr = np.asarray(all_r); c_arr = np.asarray(all_c)
        rows.append(summarize(cfg, controller.name, r_arr, c_arr, all_l, fallback_total))
        traces[controller.name] = {
            "residual_rms_time": np.sqrt(np.mean(r_arr[0] ** 2, axis=1)).tolist(),
            "latency_ms": all_l[: cfg.sequence_length],
        }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": [asdict(r) for r in rows],
        "artifacts": {
            "model_sha256": sha256_file(model_path),
            "linear_sha256": sha256_file(linear_path),
        },
        "traces": traces,
    }
    output_json.write_text(json.dumps(payload, indent=2))
    return rows
