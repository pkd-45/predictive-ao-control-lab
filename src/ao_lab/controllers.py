from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import torch

from .models import PredictorMLP


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def command(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray: ...


@dataclass
class IntegratorController:
    gain: float
    actuator_limit: float
    name: str = "integrator"

    def reset(self) -> None:
        return None

    def command(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        return np.clip(previous_command + self.gain * measurement, -self.actuator_limit, self.actuator_limit)


class HistoryPredictorBase:
    name = "predictor"

    def __init__(self, history: int, n_modes: int) -> None:
        self.history = history
        self.n_modes = n_modes
        self._hist: deque[np.ndarray] = deque(maxlen=history)

    def reset(self) -> None:
        self._hist.clear()

    def _feature(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray | None:
        # residual = atmosphere - previous_command; therefore residual + previous_command estimates atmosphere.
        x_est = measurement + previous_command
        self._hist.append(x_est.copy())
        if len(self._hist) < self.history:
            return None
        return np.stack(list(self._hist), axis=0)


class LinearPredictiveController(HistoryPredictorBase):
    name = "linear_predictor"

    def __init__(self, history: int, n_modes: int, weights: np.ndarray, bias: np.ndarray, actuator_limit: float) -> None:
        super().__init__(history, n_modes)
        self.weights = weights
        self.bias = bias
        self.actuator_limit = actuator_limit
        self.warmup = IntegratorController(gain=0.45, actuator_limit=actuator_limit)

    def command(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        feat = self._feature(measurement, previous_command)
        if feat is None:
            return self.warmup.command(measurement, previous_command)
        pred = feat.reshape(-1) @ self.weights + self.bias
        return np.clip(pred, -self.actuator_limit, self.actuator_limit)


class MLPredictiveController(HistoryPredictorBase):
    name = "ml_predictor"

    def __init__(self, history: int, n_modes: int, model: PredictorMLP, actuator_limit: float) -> None:
        super().__init__(history, n_modes)
        self.model = model.eval()
        self.actuator_limit = actuator_limit
        self.warmup = IntegratorController(gain=0.45, actuator_limit=actuator_limit)

    def command(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        feat = self._feature(measurement, previous_command)
        if feat is None:
            return self.warmup.command(measurement, previous_command)
        with torch.no_grad():
            x = torch.from_numpy(feat.astype(np.float32))[None, ...]
            pred = self.model(x).cpu().numpy()[0]
        # Deliberately leave limit checking to SafeController.
        return pred.astype(np.float64)


class OpenLoopController:
    name = "open_loop"

    def __init__(self, n_modes: int) -> None:
        self.n_modes = n_modes

    def reset(self) -> None:
        return None

    def command(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_modes, dtype=np.float64)


class SafeController:
    """Validate an ML control command and fall back to a deterministic integrator."""

    name = "ml_safe"

    def __init__(
        self,
        primary: Controller,
        fallback: IntegratorController,
        actuator_limit: float,
        latency_budget_ms: float,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.actuator_limit = actuator_limit
        self.latency_budget_ms = latency_budget_ms
        self.fallback_count = 0
        self.last_latency_ms = 0.0
        self.fallback_reasons: dict[str, int] = {}

    def reset(self) -> None:
        self.primary.reset()
        self.fallback.reset()
        self.fallback_count = 0
        self.last_latency_ms = 0.0
        self.fallback_reasons = {}

    def _fallback(self, reason: str, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        self.fallback_count += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        return self.fallback.command(measurement, previous_command)

    def command(self, measurement: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        start = perf_counter()
        try:
            command = self.primary.command(measurement, previous_command)
        # Safety boundary: any primary-controller failure triggers deterministic fallback.
        except Exception:  # noqa: BLE001
            self.last_latency_ms = (perf_counter() - start) * 1e3
            return self._fallback("exception", measurement, previous_command)
        self.last_latency_ms = (perf_counter() - start) * 1e3
        if not np.all(np.isfinite(command)):
            return self._fallback("non_finite", measurement, previous_command)
        if np.max(np.abs(command)) > self.actuator_limit:
            return self._fallback("actuator_limit", measurement, previous_command)
        if self.last_latency_ms > self.latency_budget_ms:
            return self._fallback("latency_budget", measurement, previous_command)
        return command
