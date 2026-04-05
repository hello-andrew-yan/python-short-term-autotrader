from dataclasses import dataclass

import pandas as pd

from autotrader.core.schemas import DateWindow
from autotrader.model.dataset import Dataset


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
            train=dataset.between(train),
            val=dataset.between(val),
            test=dataset.between(test),
        )

    def get_tickers(self, tickers: str | list[str]) -> "DatasetSplit":
        return DatasetSplit(
            train=self.train.get_tickers(tickers),
            val=self.val.get_tickers(tickers),
            test=self.test.get_tickers(tickers),
        )
