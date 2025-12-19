from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimulationConfig:
    save_path: Path
    cell: str
    eodf: float
    duration: float = 2
    trials: int = 100_000
    contrasts: list[float] = field(default_factory=lambda: [0.1])
    batch_size: int = 2000
    nperseg: int = 2**15
    fs: int = 30_000
    jax_key: int = 42
    wh_low: float = 0.0
    wh_high: float = 300.0
    sigma: float = 0.001
    ktime: float = 4
