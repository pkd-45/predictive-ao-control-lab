from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class AtmosphereConfig:
    rho: float
    process_sigma: float
    coupling: float


@dataclass(frozen=True)
class TrainingConfig:
    hidden: int
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class Config:
    seed: int
    n_modes: int
    sequence_length: int
    train_sequences: int
    val_sequences: int
    test_sequences: int
    history: int
    atmosphere: AtmosphereConfig
    wfs_noise_sigma: float
    actuator_limit: float
    integrator_gain: float
    latency_budget_ms: float
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return Config(
        seed=int(raw["seed"]),
        n_modes=int(raw["n_modes"]),
        sequence_length=int(raw["sequence_length"]),
        train_sequences=int(raw["train_sequences"]),
        val_sequences=int(raw["val_sequences"]),
        test_sequences=int(raw["test_sequences"]),
        history=int(raw["history"]),
        atmosphere=AtmosphereConfig(**raw["atmosphere"]),
        wfs_noise_sigma=float(raw["wfs_noise_sigma"]),
        actuator_limit=float(raw["actuator_limit"]),
        integrator_gain=float(raw["integrator_gain"]),
        latency_budget_ms=float(raw["latency_budget_ms"]),
        training=TrainingConfig(**raw["training"]),
    )
