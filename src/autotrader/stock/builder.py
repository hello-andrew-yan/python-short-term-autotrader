from collections.abc import Sequence

import pandas as pd

from autotrader.core.base import Feature, Label
from autotrader.model.dataset import Dataset
from autotrader.stock.history import StockHistory


class DatasetBuilder:
    def __init__(
        self,
        history: StockHistory,
        features: Sequence[Feature] | Feature,
        label: Label,
    ):
        self.history = history
        self.features = (
            [features] if isinstance(features, Feature) else list(features)
        )
        self.label = label

    def build(self, freq: str | None = None) -> Dataset:
        df = self.history.get_data(freq=freq)

        X = pd.concat([f(df) for f in sorted(self.features, key=str)], axis=1)
        y = self.label(df)

        X, y = X.align(y, join="inner", axis=0)
        X = X.dropna()

        return Dataset(X, y)
