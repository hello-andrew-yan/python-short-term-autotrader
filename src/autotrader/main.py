import calendar

from autotrader.core.helpers import print_metrics
from autotrader.core.history import StockHistory
from autotrader.core.types import DateWindow
from autotrader.model.dataset import Dataset
from autotrader.v1.custom.features.sma import SMASlope, SMASpread
from autotrader.v1.custom.features.time import MonthFeature
from autotrader.v1.custom.features.volume import VolumeZ
from autotrader.v1.custom.labels.forward import ForwardReturn
from autotrader.v1.model.predictor import StockPredictor

FEATURES = ["WDC", "MU"]
HELPERS = ["SPY", "SMH", "GOOGL", "LRCX"]

TRAIN_WINDOW = DateWindow.from_string("2000-01-01", "2022-12-31")
VAL_WINDOW = DateWindow.from_string("2023-01-01", "2024-12-31")
TEST_WINDOW = DateWindow.from_string("2025-01-01", "2025-12-31")


def main() -> None:
    # logging.basicConfig(level=logging.INFO, format="(%(name)s) %(message)s")
    dataset = Dataset.from_history(
        history=StockHistory(
            tickers=[*FEATURES, *HELPERS],
            window=DateWindow(TRAIN_WINDOW.start, TEST_WINDOW.end),
        ),
        features=[
            SMASpread(short=20, long=200),
            SMASlope(period=20),
            SMASlope(period=200),
            VolumeZ(period=20),
            VolumeZ(period=200),
            MonthFeature(
                focus_months=[
                    calendar.JANUARY,
                    calendar.JUNE,
                    calendar.OCTOBER,
                    calendar.NOVEMBER,
                ]
            ),
        ],
        label=ForwardReturn(gain_threshold=0.01),
    )

    train = dataset.between(TRAIN_WINDOW)
    val = dataset.between(VAL_WINDOW)

    predictor = StockPredictor()
    predictor.fit(train, val)

    test_result = predictor.eval(
        dataset.between(TEST_WINDOW),
        min_confidence=0.5,
    )
    print_metrics(
        test_result,
        min_precision=0.60,
        sort_by=["Precision", "Total"],
        ascending=False,
    )


if __name__ == "__main__":
    main()
