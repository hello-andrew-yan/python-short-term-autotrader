# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
from dataclasses import dataclass


@dataclass
class PredictorConfig:
    # Model
    objective: str = "binary:logistic"
    n_estimators: int = 4000
    max_depth: int = 5

    # Learning
    learning_rate: float = 0.001
    early_stopping_rounds: int = 50

    # Regularisation
    gamma: float = 0.3
    min_child_weight: int = 5
    reg_alpha: float = 3.0
    reg_lambda: float = 2.0

    # Sampling
    subsample: float = 0.7
    colsample_bytree: float = 0.7

    # Hardware
    nthread: int = -1
    tree_method: str = "hist"
    device: str = "cpu"

    # Reproducibility
    random_state: int = 42
