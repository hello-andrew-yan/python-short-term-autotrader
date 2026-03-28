from collections.abc import Sequence

import pandas as pd

from autotrader import logger
from autotrader.core.base import Feature, Label
from autotrader.core.data import Dataset, DatasetSplit
from autotrader.core.stock import StockHistory


class StockDatasetBuilder:
    def __init__(
        self,
        history: StockHistory,
        features: Sequence[Feature],
        label: Label,
    ):
        self.history = history
        self.features = features
        self.label = label
        self._dataset: Dataset | None = None

    def __call__(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self._build()
        return self._dataset

    def _validate_features(self, df_features: pd.DataFrame) -> None:
        if df_features.columns.duplicated().any():
            dupes = df_features.columns[
                df_features.columns.duplicated()
            ].tolist()
            raise ValueError(f"Duplicate feature columns: {dupes}")

        if "Label" in df_features.columns:
            raise ValueError(
                "Feature column named 'Label' conflicts with label column."
            )

    def _build(self) -> Dataset:
        df = self.history.get_data()
        initial_rows = len(df)

        df_features = pd.concat(
            [feature(df) for feature in self.features], axis=1
        )
        self._validate_features(df_features)

        labels = self.label(df)

        combined = pd.concat([df_features, labels.to_frame("Label")], axis=1)
        combined = combined.sort_index().dropna()

        rows_lost = initial_rows - len(combined)
        if rows_lost > 0:
            logger.info("Dropped %d incomplete rows", rows_lost)

        X = combined[df_features.columns]
        y = combined["Label"].astype(int)

        logger.info(
            "Final dataset: %d rows, %d features", len(X), len(X.columns)
        )

        return Dataset(X, y)

    def _validate_split(
        self,
        train_range: tuple[str, str],
        val_range: tuple[str, str],
        test_range: tuple[str, str],
    ) -> None:
        timestamps = [
            (pd.Timestamp(s), pd.Timestamp(e))
            for s, e in (train_range, val_range, test_range)
        ]
        names = ["train", "val", "test"]

        for name, (start, end) in zip(names, timestamps, strict=True):
            if start > end:
                raise ValueError(f"{name}_start must be before {name}_end")

        for i in range(len(timestamps) - 1):
            if timestamps[i][1] >= timestamps[i + 1][0]:
                raise ValueError(
                    f"{names[i]}_end must be before {names[i + 1]}_start"
                )

    def split(
        self,
        train_range: tuple[str, str],
        val_range: tuple[str, str],
        test_range: tuple[str, str],
    ) -> DatasetSplit:
        self._validate_split(train_range, val_range, test_range)
        dataset = self()

        return DatasetSplit(
            train=dataset.filter_by_date(*train_range),
            val=dataset.filter_by_date(*val_range),
            test=dataset.filter_by_date(*test_range),
        )
