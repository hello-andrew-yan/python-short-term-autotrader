import logging

from autotrader.core.schema import DateWindow
from autotrader.data.history import StockHistory
from autotrader.extension.custom import SMA, ForwardReturn
from autotrader.ml import DatasetBuilder, DatasetSplit, StockPredictor

FEATURES = ["LITE", "MU"]
HELPERS = ["SPY", "NXT", "ARM", "AMZN", "MRVL"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="(%(name)s) %(message)s")

    builder = DatasetBuilder(
        StockHistory(tickers=[*FEATURES, *HELPERS]),
        SMA(),
        ForwardReturn(gain_threshold=0.015),
    )
    dataset = builder.build()
    split = DatasetSplit.from_dates(
        dataset,
        train=DateWindow("1980-01-01", "2024-06-30"),
        val=DateWindow("2024-07-01", "2025-06-30"),
        test=DateWindow("2025-07-01", "2026-03-05"),
    )

    predictor = StockPredictor()
    predictor.train(split.train, split.val, verbose=True)
    predictor.test(split.test, ignore=HELPERS)


if __name__ == "__main__":
    main()
