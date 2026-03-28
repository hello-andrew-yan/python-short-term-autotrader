from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class _Base(ABC):
    # Column and index constants based on the
    # resampled DataFrame from StockHistory

    DATE = "Date"
    TICKER = "Ticker"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"

    @property
    @abstractmethod
    def required_columns(self) -> Sequence[str]: ...

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [
            col for col in self.required_columns if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} missing required columns: {missing}"
            )

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series: ...

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        self._validate(df)
        return self._calculate(df)


class Feature(_Base):
    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame: ...


class Label(_Base):
    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series: ...

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        self._validate(df)
        return self._calculate(df)
