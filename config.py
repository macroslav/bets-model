from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_estimators: int
    learning_rate: int
    verbose: int = 0
