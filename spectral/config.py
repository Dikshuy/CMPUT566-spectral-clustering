from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    """two moons"""
    n: int = 2000
    noise: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class GraphConfig:
    """kNN graph with Gaussian weights"""
    k: int = 10


@dataclass(frozen=True)
class NetConfig:
    """MLP: (Linear + relu)^{hid_layers} -> Linear"""
    hid_dim: int = 256
    hid_layers: int = 2


@dataclass(frozen=True)
class ALLOConfig:
    """ALLO: Laplacian eigenvectors with learned duals and barrier"""
    d: int = 8
    net: NetConfig = NetConfig()
    lr: float = 1e-3
    barrier_init: float = 0.5
    barrier_min: float = 0.0
    barrier_max: float = 5.0
    duals_init: float = -1.0
    duals_min: float = -100.0
    duals_max: float = 100.0
    duals_step_size: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    steps: int = 10_000
    batch_size: int = 1024
    log_every: int = 1000


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    allo: ALLOConfig = field(default_factory=ALLOConfig)
    run: RunConfig = field(default_factory=RunConfig)
