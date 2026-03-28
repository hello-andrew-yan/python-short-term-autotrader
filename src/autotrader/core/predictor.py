import logging

import pandas as pd
import xgboost as xgb

from autotrader import logger
from autotrader.core.dataset import Dataset


class StockPredictor:
    DEFAULT_CONFIG = dict(
        nthread=-1,
        n_estimators=2000,
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

    def __init__(
        self,
        config: dict | None = None,
    ):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] | None = None

    def _log_feature_importance(self) -> None:
        if not self.model or not self.feature_names:
            return

        importance = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)
        logger.info("Model Feature Importance:\n%s", importance)

    def _log_trades(self, trades: pd.DataFrame, min_confidence: float) -> None:
        logger.info("Confidence threshold: %.2f%%", min_confidence * 100)
        for ticker, row in trades.iterrows():
            logger.info(
                "%s | Trades: %d (W:%d L:%d) | Precision: %.2f%%",
                ticker,
                row["Total"],
                row["Wins"],
                row["Losses"],
                row["Precision"] * 100,
            )

    def train(
        self,
        train: Dataset,
        val: Dataset,
        verbose: bool = False,
    ) -> "StockPredictor":
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        logger.info(f"Training on {len(train.X)} weekly samples")

        self.feature_names = train.X.columns.tolist()

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
        logger.info(f"Best Iteration: {self.model.best_iteration}")
        self._log_feature_importance()

        return self

    def test(
        self,
        test: Dataset,
        min_confidence: float = 0.5,
        ignore: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model must be trained before testing.")

        logger.info("Testing on %d samples", len(test.X))

        self.model.set_params(device="cpu")
        probs = pd.Series(
            self.model.predict_proba(test.X)[:, 1],
            index=test.X.index,
        )

        all_tickers = test.y.index.get_level_values("Ticker").unique()
        if ignore:
            all_tickers = [t for t in all_tickers if t not in ignore]

        mask = probs >= min_confidence
        y_filtered = test.y.loc[
            test.y.index.get_level_values("Ticker").isin(all_tickers)
        ]

        trades = (
            y_filtered[mask]
            .groupby(level="Ticker")
            .agg(Total="count", Wins="sum")
            .reindex(all_tickers, fill_value=0)
            .assign(
                Losses=lambda x: x["Total"] - x["Wins"],
                Precision=lambda x: (x["Wins"] / x["Total"]).fillna(0.0),
            )
            .astype({"Total": int, "Wins": int, "Losses": int})
            .sort_values("Total", ascending=False)
        )

        self._log_trades(trades, min_confidence)
        return trades
