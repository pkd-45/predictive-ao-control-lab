import numpy as np

from ao_lab.atmosphere import generate_phase_sequence
from ao_lab.config import load_config
from ao_lab.controllers import IntegratorController, OpenLoopController, SafeController
from ao_lab.evaluate import run_episode


class BadController:
    name = "bad"
    def reset(self): pass
    def command(self, measurement, previous_command):
        return np.full_like(previous_command, np.nan)


class HugeController:
    name = "huge"
    def reset(self): pass
    def command(self, measurement, previous_command):
        return np.full_like(previous_command, 1e6)


def test_integrator_improves_on_open_loop():
    cfg = load_config("configs/baseline.yaml")
    phase = generate_phase_sequence(cfg, 999)
    r0, _, _ = run_episode(cfg, phase, OpenLoopController(cfg.n_modes), 1000)
    ri, _, _ = run_episode(cfg, phase, IntegratorController(cfg.integrator_gain, cfg.actuator_limit), 1000)
    assert np.sqrt(np.mean(ri[20:] ** 2)) < np.sqrt(np.mean(r0[20:] ** 2))


def test_nonfinite_ml_falls_back():
    cfg = load_config("configs/baseline.yaml")
    safe = SafeController(BadController(), IntegratorController(cfg.integrator_gain, cfg.actuator_limit), cfg.actuator_limit, 1000.0)
    y = np.ones(cfg.n_modes)
    u = safe.command(y, np.zeros(cfg.n_modes))
    assert np.all(np.isfinite(u))
    assert safe.fallback_count == 1
    assert safe.fallback_reasons["non_finite"] == 1


def test_actuator_limit_violation_falls_back():
    cfg = load_config("configs/baseline.yaml")
    safe = SafeController(HugeController(), IntegratorController(cfg.integrator_gain, cfg.actuator_limit), cfg.actuator_limit, 1000.0)
    y = np.ones(cfg.n_modes)
    u = safe.command(y, np.zeros(cfg.n_modes))
    assert np.max(np.abs(u)) <= cfg.actuator_limit
    assert safe.fallback_reasons["actuator_limit"] == 1
