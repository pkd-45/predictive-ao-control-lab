from __future__ import annotations

from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from .atmosphere import generate_phase_sequence
from .config import Config


def _seeds(cfg: Config, split: str) -> list[int]:
    counts = {"train": cfg.train_sequences, "val": cfg.val_sequences, "test": cfg.test_sequences}
    offsets = {"train": 0, "val": 100_000, "test": 200_000}
    return [cfg.seed + offsets[split] + i for i in range(counts[split])]


def generate_dataset(cfg: Config, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.attrs["seed"] = cfg.seed
        h5.attrs["n_modes"] = cfg.n_modes
        h5.attrs["sequence_length"] = cfg.sequence_length
        for split in ("train", "val", "test"):
            seeds = _seeds(cfg, split)
            phases = np.stack([generate_phase_sequence(cfg, s) for s in seeds], axis=0)
            grp = h5.create_group(split)
            grp.create_dataset("phase", data=phases, compression="gzip")
            grp.create_dataset("seed", data=np.asarray(seeds, dtype=np.int64))
    return output


def load_split(path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[split]["phase"]), np.asarray(h5[split]["seed"])


def make_windows(phases: np.ndarray, history: int, noise_sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build noisy-history -> next-clean-state supervised examples."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for seq in phases:
        noisy = seq + rng.normal(0.0, noise_sigma, size=seq.shape)
        for t in range(history, len(seq) - 1):
            xs.append(noisy[t - history + 1 : t + 1])
            ys.append(seq[t + 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
