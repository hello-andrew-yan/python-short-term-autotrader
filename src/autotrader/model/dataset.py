from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd
from pandera.typing import DataFrame

from autotrader.core.base import Feature, Label
from autotrader.core.history import StockHistory
from autotrader.core.schemas import StockPriceIndex
from autotrader.core.types import DateWindow

type Indexer = pd.Timestamp | slice | Sequence[str]


@dataclass(frozen=True)
class Dataset:
    X: DataFrame[StockPriceIndex]
    y: pd.Series

    def __post_init__(self) -> None:
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y indices must match")

        if not self.X.index.is_monotonic_increasing:
            object.__setattr__(self, "X", self.X.sort_index())
            object.__setattr__(self, "y", self.y.sort_index())

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
            [features] if isinstance(features, Feature) else list(features)
        )

        X = pd.concat([f(df) for f in sorted(feature_list, key=str)], axis=1)
        y = label(df)

        X, y = X.align(y, join="inner", axis=0)
        X = X.dropna()
        y = y.loc[X.index]

        return cls(cast(DataFrame[StockPriceIndex], X), y)

    def _slice(
        self, ticker_idx: Indexer = slice(None), date_idx: Indexer = slice(None)
    ) -> "Dataset":
        key = (ticker_idx, date_idx)

        return Dataset(
            cast(DataFrame[StockPriceIndex], self.X.loc[key, :]),
            self.y.loc[key],
        )

    def between(self, window: DateWindow) -> "Dataset":
        return self._slice(date_idx=slice(window.start, window.end))

    def ticker(self, tickers: str | Sequence[str]) -> "Dataset":
        ticker_list = (
            [tickers] if isinstance(tickers, str) else sorted(list(tickers))
        )
        return self._slice(ticker_idx=ticker_list)
