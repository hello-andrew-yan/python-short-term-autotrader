import logging
from dataclasses import asdict, dataclass

import pandas as pd
import xgboost as xgb
from pandera.typing import DataFrame

from autotrader import logger
from autotrader.v1.core.schemas import TradeResult
from autotrader.v1.model.dataset import Dataset


# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
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


class StockPredictor:
    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
        self.model = None

    def train(
        self, train: Dataset, val: Dataset | None = None
    ) -> "StockPredictor":
        p = train.y.mean()
        weight = (1 - p) / p if 0 < p < 1 else 1.0

        logger.info(
            f"samples={len(train.X)}, balance={p:.2%}, weight={weight:.2f}"
        )

        params = asdict(self.config)
        stopping = params.pop("early_stopping_rounds") if val else None

        self.model = xgb.XGBClassifier(
            **params, scale_pos_weight=weight, early_stopping_rounds=stopping
        )

        self.model.fit(
            train.X,
            train.y,
            eval_set=[(val.X, val.y)] if val else None,
            verbose=100
            if logger.getEffectiveLevel() <= logging.INFO
            else False,
        )

        self.importance = pd.Series(
            self.model.feature_importances_, index=train.X.columns
        ).sort_values(ascending=False)

        logger.info(
            f"importance=\n{self.importance.sort_values(ascending=False)}"
        )
        return self

    def test(
        self,
        test: Dataset,
        min_confidence: float = 0.5,
    ) -> DataFrame[TradeResult]:
        if self.model is None:
            raise ValueError("Model must be trained before testing.")

        probs = pd.Series(
            self.model.predict_proba(test.X)[:, 1], index=test.X.index
        )
        stats = (
            test.y[probs >= min_confidence]
            .groupby(level="Ticker")
            .agg(Total="count", Wins="sum")
            .assign(
                Losses=lambda df: df.Total - df.Wins,
                Precision=lambda df: (df.Wins / df.Total).fillna(0.0),
            )
            .sort_values("Total", ascending=False)
        )

        return TradeResult.validate(stats)
