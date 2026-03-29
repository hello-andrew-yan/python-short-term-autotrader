from dataclasses import dataclass

import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self):
        if not isinstance(self.X.index, pd.MultiIndex):
            raise ValueError("X must have a MultiIndex")

        if self.X.index.names != ["Ticker", "Date"]:
            raise ValueError(
                "Index must be MultiIndex with levels ['Ticker', 'Date']. "
                f"Found: {self.X.index.names}"
            )

        if not self.X.index.equals(self.y.index):
            raise ValueError("X and y indices must match")

    def __repr__(self) -> str:
        return f"Dataset(rows={len(self.X):,}, features={self.X.shape[1]:,})"

    def filter_by_date(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "Dataset":
        t_start = pd.Timestamp(start) if start is not None else None
        t_end = pd.Timestamp(end) if end is not None else None

        if t_start is not None and t_end is not None and t_start > t_end:
            raise ValueError(f"Start {t_start} must be before end {t_end}")

        dates = self.X.index.get_level_values("Date")
        mask = pd.Series(True, index=self.X.index)

        if t_start is not None:
            mask &= dates >= t_start
        if t_end is not None:
            mask &= dates <= t_end

        return Dataset(self.X[mask], self.y[mask])

    def filter_by_ticker(self, tickers: str | list[str]) -> "Dataset":
        if isinstance(tickers, str):
            tickers = [tickers]

        mask = self.X.index.get_level_values("Ticker").isin(tickers)
        return Dataset(self.X[mask], self.y[mask])


@dataclass
class DatasetSplit:
    train: Dataset
    val: Dataset
    test: Dataset

    def __repr__(self) -> str:
        return (
            f"DatasetSplit(train={len(self.train.X):,}, "
            f"val={len(self.val.X):,}, "
            f"test={len(self.test.X):,})"
        )

    def filter_by_ticker(self, tickers: str | list[str]) -> "DatasetSplit":
        return DatasetSplit(
            train=self.train.filter_by_ticker(tickers),
            val=self.val.filter_by_ticker(tickers),
            test=self.test.filter_by_ticker(tickers),
        )
