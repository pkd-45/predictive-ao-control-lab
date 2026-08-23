import numpy as np

from ao_lab.atmosphere import generate_phase_sequence
from ao_lab.config import load_config


def test_reproducible_seed():
    cfg = load_config("configs/baseline.yaml")
    a = generate_phase_sequence(cfg, 123)
    b = generate_phase_sequence(cfg, 123)
    assert np.array_equal(a, b)
