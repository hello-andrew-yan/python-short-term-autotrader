from dataclasses import dataclass
from typing import cast

import pandas as pd

from autotrader.core.schemas import DateWindow, StockPriceIndex


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self) -> None:
        StockPriceIndex.validate(self.X)
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y indices must match")

    def between(self, window: DateWindow) -> "Dataset":
        dates = pd.to_datetime(
            self.X.index.get_level_values(StockPriceIndex.Date)
        )
        mask = (dates >= window.start if window.start else True) & (
            dates <= window.end if window.end else True
        )
        return Dataset(cast(pd.DataFrame, self.X.loc[mask]), self.y.loc[mask])

    def tickers(self, tickers: str | list[str]) -> "Dataset":
        _tickers = [tickers] if isinstance(tickers, str) else tickers
        mask = self.X.index.get_level_values(StockPriceIndex.Ticker).isin(
            _tickers
        )

        return Dataset(self.X.loc[mask], self.y.loc[mask])
