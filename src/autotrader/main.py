import calendar

from rich import print as rprint

from autotrader.v1.core.helpers import print_results
from autotrader.v1.core.schemas import DateWindow
from autotrader.v1.custom.features.sma import SMASlope, SMASpread
from autotrader.v1.custom.features.time import MonthFeature
from autotrader.v1.custom.features.volume import VolumeZ
from autotrader.v1.custom.labels.forward import ForwardReturn
from autotrader.v1.model.dataset import Dataset
from autotrader.v1.model.history import StockHistory
from autotrader.v1.model.predictor import StockPredictor

FEATURES = ["WDC", "MU"]
HELPERS = ["SPY", "SMH", "GOOGL", "LRCX"]

TRAIN_WINDOW = DateWindow.from_string("2000-01-01", "2022-12-31")
VAL_WINDOW = DateWindow.from_string("2023-01-01", "2024-12-31")
TEST_WINDOW = DateWindow.from_string("2025-01-01", "2025-12-31")
LIVE_WINDOW = DateWindow.from_string("2026-01-01", "2026-12-31")


def main() -> None:
    # logging.basicConfig(level=logging.INFO, format="(%(name)s) %(message)s")
    dataset = Dataset.from_history(
        history=StockHistory(
            tickers=[*FEATURES, *HELPERS],
            window=DateWindow(TRAIN_WINDOW.start, LIVE_WINDOW.end),
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
    predictor.train(train, val)

    test_result = predictor.test(
        dataset.between(TEST_WINDOW),
        min_confidence=0.5,
    )
    live_result = predictor.test(
        dataset.between(LIVE_WINDOW),
        min_confidence=0.5,
    )

    rprint(
        f"Test Window: {TEST_WINDOW.start.date()} to {TEST_WINDOW.end.date()}"
    )
    print_results(test_result, min_precision=0.6)

    rprint(
        f"Live Window: {LIVE_WINDOW.start.date()} to {LIVE_WINDOW.end.date()}"
    )
    print_results(live_result, min_precision=0.6)


if __name__ == "__main__":
    main()
