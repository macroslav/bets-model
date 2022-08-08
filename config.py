from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_estimators: int
    learning_rate: int
    verbose: int = 0
