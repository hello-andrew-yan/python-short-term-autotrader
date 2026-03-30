from dataclasses import dataclass
from typing import cast

import pandas as pd

from autotrader.core.schema import DateWindow
from autotrader.core.schema import HistoryIndex as H


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self) -> None:
        self.X = self.X.sort_index(axis=0).sort_index(axis=1)
        self.y = self.y.sort_index()

        H.validate(self.X)
        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y indices must match")

    def between(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "Dataset":
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None

        dates = pd.to_datetime(self.X.index.get_level_values(H.Date))
        mask = (dates >= start_ts if start_ts else True) & (
            dates <= end_ts if end_ts else True
        )

        return Dataset(cast(pd.DataFrame, self.X.loc[mask]), self.y.loc[mask])

    def get_tickers(self, tickers: str | list[str]) -> "Dataset":
        _tickers = [tickers] if isinstance(tickers, str) else tickers
        mask = self.X.index.get_level_values(H.Ticker).isin(_tickers)

        return Dataset(self.X.loc[mask], self.y.loc[mask])


@dataclass
class DatasetSplit:
    train: Dataset
    val: Dataset
    test: Dataset

    @classmethod
    def from_dates(
        cls,
        dataset: Dataset,
        train: DateWindow,
        val: DateWindow,
        test: DateWindow,
    ) -> "DatasetSplit":
        boundaries = [
            pd.Timestamp(d) for window in (train, val, test) for d in window
        ]
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("All date boundaries must be unique")
        if boundaries != sorted(boundaries):
            raise ValueError("Dates must be in chronological order")

        return cls(
            train=dataset.between(*train),
            val=dataset.between(*val),
            test=dataset.between(*test),
        )

    def get_tickers(self, tickers: str | list[str]) -> "DatasetSplit":
        return DatasetSplit(
            train=self.train.get_tickers(tickers),
            val=self.val.get_tickers(tickers),
            test=self.test.get_tickers(tickers),
        )
