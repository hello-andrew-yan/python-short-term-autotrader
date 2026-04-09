from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from autotrader.v1.core.base import Feature, Label
from autotrader.v1.core.schemas import DateWindow, StockPriceIndex
from autotrader.v1.model.history import StockHistory


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self) -> None:
        StockPriceIndex.validate(self.X)
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y indices must match")

    @classmethod
    def from_history(
        cls,
        history: StockHistory,
        features: Sequence[Feature] | Feature,
        label: Label,
        freq: str | None = None,
    ) -> "Dataset":
        df = history.get_data(freq=freq)
        feature_list = (
            [features]
            if not isinstance(features, (list, tuple, Sequence))
            else list(features)
        )

        X = pd.concat([f(df) for f in sorted(feature_list, key=str)], axis=1)
        y = label(df)

        X, y = X.align(y, join="inner", axis=0)
        X = X.dropna()
        y = y.loc[X.index]

        return cls(X, y)

    def between(self, window: DateWindow) -> "Dataset":
        dates = pd.to_datetime(self.X.index.get_level_values("Date"))
        mask = (dates >= window.start if window.start else True) & (
            dates <= window.end if window.end else True
        )
        return Dataset(cast(pd.DataFrame, self.X.loc[mask]), self.y.loc[mask])

    def tickers(self, tickers: str | list[str]) -> "Dataset":
        _tickers = [tickers] if isinstance(tickers, str) else tickers
        mask = self.X.index.get_level_values("Ticker").isin(_tickers)

        return Dataset(self.X.loc[mask], self.y.loc[mask])
