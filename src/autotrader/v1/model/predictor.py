import logging
from dataclasses import asdict

import pandas as pd
import xgboost as xgb
from pandera.typing import DataFrame

from autotrader import logger
from autotrader.core.schemas import PerformanceMetrics as PM
from autotrader.core.schemas import StockPriceIndex
from autotrader.model.dataset import Dataset
from autotrader.v1.model.config import PredictorConfig


class StockPredictor:
    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
        self.model: xgb.XGBClassifier | None = None
        self.importance: pd.Series | None = None

    def fit(
        self, train: Dataset, val: Dataset | None = None
    ) -> "StockPredictor":
        p = train.y.mean()
        weight = (1 - p) / p if 0 < p < 1 else 1.0

        logger.info(
            f"Training: samples={len(train.X)}, "
            f"balance={p:.2%}, weight={weight:.2f}"
        )

        cfg = asdict(self.config)
        stop = cfg.pop("early_stopping_rounds") if val else None

        self.model = xgb.XGBClassifier(
            **cfg, scale_pos_weight=weight, early_stopping_rounds=stop
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

        return self

    def eval(self, test: Dataset, min_confidence: float = 0.5) -> DataFrame[PM]:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        probs = pd.Series(
            self.model.predict_proba(test.X)[:, 1], index=test.X.index
        )
        stats = (
            test.y[probs >= min_confidence]
            .groupby(level=StockPriceIndex.Ticker)
            .agg(Total="count", Wins="sum", Precision="mean")
            .assign(Losses=lambda df: df.Total - df.Wins)
            .fillna(0.0)
        )
        # Enforcing column order from schema
        cols = list(PM.to_schema().columns.keys())

        return PM.validate(stats[cols])
