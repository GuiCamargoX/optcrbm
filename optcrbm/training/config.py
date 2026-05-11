"""Training configuration for examples."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 100
    write_outputs: bool = False
    results_dir: str = "./results"
