from typing import cast

import pandas as pd
import xgboost as xgb
from pandera.typing import DataFrame

from autotrader import logger
from autotrader.core.schema import HistoryFrame as F
from autotrader.core.schema import TradeResult as T
from autotrader.ml.dataset import Dataset

TEMP_DEFAULT_CONFIG = dict(
    nthread=1,
    n_estimators=2500,
    max_depth=5,
    learning_rate=0.0025,
    objective="binary:logistic",
    gamma=0.3,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=3,
    reg_lambda=2,
    random_state=42,
    early_stopping_rounds=25,
)


class StockPredictor:
    def __init__(self, config: dict | None = None):
        self.config = config or TEMP_DEFAULT_CONFIG
        self.model: xgb.XGBClassifier | None = None

    def _log_trades(self, trades: DataFrame[T], min_confidence: float) -> None:
        logger.info("Confidence threshold: %.2f%%", min_confidence * 100)
        for ticker, row in trades.iterrows():
            logger.info(
                "%s | Trades: %d (W:%d L:%d) | Precision: %.2f%%",
                ticker,
                row[T.Total],
                row[T.Wins],
                row[T.Losses],
                row[T.Precision] * 100,
            )

    def train(
        self, train: Dataset, val: Dataset, verbose: bool = False
    ) -> "StockPredictor":
        logger.info(f"Training on {len(train.X)} weekly samples")

        p = train.y.mean()
        self.config.setdefault("scale_pos_weight", (1 - p) / p if p else 1.0)
        logger.info(
            f"Class balance: {p:>7.2%}, "
            f"weight: {self.config['scale_pos_weight']:.2f}"
        )

        self.model = xgb.XGBClassifier(**self.config).fit(
            train.X,
            train.y,
            eval_set=[(val.X, val.y)],
            verbose=100 if verbose else False,
        )
        logger.info("Best iteration: %d", self.model.best_iteration)

        importance = pd.Series(
            self.model.feature_importances_, index=train.X.columns
        ).sort_values(ascending=False)
        logger.info("Feature importance:\n%s", importance)

        return self

    def test(
        self,
        test: Dataset,
        min_confidence: float = 0.5,
        ignore: list[str] | None = None,
    ) -> DataFrame[T]:
        if self.model is None:
            raise ValueError("Model must be trained before testing.")

        logger.info(
            "Testing on %d samples | Confidence: %.2f%%",
            len(test.X),
            min_confidence * 100,
        )
        probs = pd.Series(
            self.model.predict_proba(test.X)[:, 1], index=test.X.index
        )
        y = (
            test.y
            if not ignore
            else test.y[~test.y.index.get_level_values(F.Ticker).isin(ignore)]
        )

        result = cast(
            DataFrame[T],
            (
                y[probs >= min_confidence]
                .groupby(level=F.Ticker)
                .agg(["count", "sum"])
                .rename(columns={"count": T.Total, "sum": T.Wins})
                .assign(
                    **{
                        T.Losses: lambda x: x[T.Total] - x[T.Wins],
                        T.Precision: lambda x: (x[T.Wins] / x[T.Total]).fillna(
                            0.0
                        ),
                    }
                )
                .astype({T.Total: int, T.Wins: int, T.Losses: int})
                .sort_values(T.Total, ascending=False)
            ),
        )

        logger.info("Ignored: %s", ignore)
        self._log_trades(result, min_confidence)

        return result
