import calendar

from rich import print as rprint

from autotrader.core.helpers import print_metrics
from autotrader.core.history import StockHistory
from autotrader.core.types import DateWindow
from autotrader.model.dataset import Dataset
from autotrader.v1.custom.features.sma import SMASlope, SMASpread
from autotrader.v1.custom.features.time import MonthFeature
from autotrader.v1.custom.features.volume import VolumeZ
from autotrader.v1.custom.labels.open import WeekOpenReturn
from autotrader.v1.model.predictor import StockPredictor

FEATURES = ["WDC", "MU", "LRCX"]
HELPERS = ["SPY", "SMH"]


# 1980–2022
TRAIN_WINDOW = DateWindow.from_string("1960-01-01", "2022-12-31")

# 2023-2024
# https://curvo.eu/backtest/en/market-index/nasdaq-global-artificial-intelligence-and-big-data?currency=eur
#
# Performance metrics for the AI sector in 2023 (+62.1%) and 2024 (+35.1%)
# isolates high-conviction signals within the AI-driven surge.
VAL_WINDOW = DateWindow.from_string("2023-01-01", "2024-12-31")

# 2025–2026
TEST_WINDOW = DateWindow.from_string("2025-01-01", "2026-12-31")

MIN_CONFIDENCE = 0.50


def main() -> None:
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
                    calendar.FEBRUARY,
                    calendar.MARCH,
                    calendar.APRIL,
                    calendar.MAY,
                    calendar.JUNE,
                    calendar.JULY,
                    calendar.AUGUST,
                    calendar.SEPTEMBER,
                    calendar.OCTOBER,
                    calendar.NOVEMBER,
                    calendar.DECEMBER,
                ]
            ),
        ],
        label=WeekOpenReturn(gain_threshold=0.01),
    )

    train = dataset.between(TRAIN_WINDOW)
    val = dataset.between(VAL_WINDOW)

    predictor = StockPredictor()
    predictor.fit(train, val)

    if predictor.model is None:
        raise ValueError("Model was unexpectedly not trained")

    rprint(f"best_iteration={predictor.model.best_iteration}")
    rprint(f"best_score={predictor.model.best_score:.5f}")
    rprint(f"importance=\n{predictor.importance}")
    rprint(f"train_window={TRAIN_WINDOW}")
    rprint(f"val_window={VAL_WINDOW}")
    rprint(f"test_window={TEST_WINDOW}")
    rprint(f"min_confidence={MIN_CONFIDENCE}")

    result = predictor.eval(
        dataset.between(TEST_WINDOW),
        min_confidence=MIN_CONFIDENCE,
    )
    print_metrics(
        result,
        min_precision=0.60,
        sort_by=["Precision", "Total"],
        ascending=False,
    )


if __name__ == "__main__":
    main()
