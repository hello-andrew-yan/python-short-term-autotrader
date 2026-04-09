# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
from dataclasses import dataclass


@dataclass
class PredictorConfig:
    # Model
    objective: str = "binary:logistic"
    n_estimators: int = 1000
    max_depth: int = 6

    # Learning
    learning_rate: float = 0.001
    early_stopping_rounds: int = 25

    # Regularisation
    gamma: float = 0.5  # Minimum loss reduction to split
    min_child_weight: int = 3  # Minimum weight needed per leaf
    reg_alpha: float = 1.0  # L1 regularisation for pruning
    reg_lambda: float = 1.0  # L2 regularisation for smoothing

    # Sampling
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Hardware
    nthread: int = -1
    tree_method: str = "hist"
    device: str = "cpu"

    # Reproducibility
    random_state: int = 42
