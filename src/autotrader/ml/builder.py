from collections.abc import Sequence

import pandas as pd

from autotrader.core.base import Feature, Label
from autotrader.data.history import StockHistory
from autotrader.ml.dataset import Dataset


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
        self.dataset: Dataset | None = None

    def _validate(self, X: pd.DataFrame, y: pd.Series) -> None:
        if dupes := X.columns[X.columns.duplicated()].tolist():
            raise ValueError(f"Duplicate feature columns: {dupes}")
        if y.name in X.columns:
            raise ValueError(
                f"Label '{y.name}' conflicts with a feature column"
            )

    def _build(self, verbose: bool = False) -> Dataset:
        df = self.history(verbose=verbose)
        X = pd.concat([feature(df) for feature in self.features], axis=1)
        y = self.label(df)
        self._validate(X, y)
        X, y = X.align(y, join="inner", axis=0)
        print(f"X rows: {len(X)}, y rows: {len(y)}")
        return Dataset(X, y)

    def build(self, refresh: bool = False) -> Dataset:
        if self.dataset is None or refresh:
            self.dataset = self._build()
        return self.dataset
