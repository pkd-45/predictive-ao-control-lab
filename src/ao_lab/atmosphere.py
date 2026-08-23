from __future__ import annotations

import numpy as np

from .config import Config


def generate_phase_sequence(cfg: Config, seed: int) -> np.ndarray:
    """Generate a coupled AR(1) modal phase sequence.

    This is intentionally a compact synthetic plant, not a physical METIS/ELT model.
    """
    rng = np.random.default_rng(seed)
    n_t, n_m = cfg.sequence_length, cfg.n_modes
    x = np.zeros((n_t, n_m), dtype=np.float64)
    x[0] = rng.normal(0.0, cfg.atmosphere.process_sigma, size=n_m)
    rho = cfg.atmosphere.rho
    sigma = cfg.atmosphere.process_sigma
    c = cfg.atmosphere.coupling
    for t in range(1, n_t):
        neigh = 0.5 * (np.roll(x[t - 1], 1) + np.roll(x[t - 1], -1))
        innovation = rng.normal(0.0, sigma, size=n_m)
        x[t] = rho * x[t - 1] + c * neigh + innovation
    return x
