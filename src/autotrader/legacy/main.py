import logging

from autotrader.data.history import StockHistory
from autotrader.legacy.builder import StockDatasetBuilder
from autotrader.legacy.extension.malyarovich import (
    ForwardReturn,
    MalyarovichSMA,
)
from autotrader.legacy.predictor import StockPredictor

FEATURES = ["LITE", "MU"]
HELPERS = ["ARM", "NXT", "SPY"]


def main() -> None:

    logging.basicConfig(level=logging.INFO, format="(%(name)s) %(message)s")

    history = StockHistory(tickers=[*FEATURES, *HELPERS])

    features = [MalyarovichSMA()]
    label = ForwardReturn(gain_threshold=0.015)

    builder = StockDatasetBuilder(history, features=features, label=label)
    split = builder.split(
        train_range=("1980-01-01", "2024-06-30"),
        val_range=("2024-07-01", "2025-06-30"),
        test_range=("2025-07-01", "2026-03-05"),
    )

    predictor = StockPredictor()
    predictor.train(split.train, split.val, verbose=True)
    predictor.test(split.test, ignore=HELPERS)


if __name__ == "__main__":
    main()
