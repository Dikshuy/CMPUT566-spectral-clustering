import numpy as np
from sklearn import datasets

from spectral.config import DataConfig


def make_moons(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """(x, labels), x as float32 of shape (n, 2)"""
    x, y = datasets.make_moons(cfg.n, noise=cfg.noise, random_state=cfg.seed)
    return x.astype(np.float32), y.astype(np.int32)
