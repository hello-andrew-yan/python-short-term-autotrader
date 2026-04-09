from abc import ABC, abstractmethod

import pandas as pd
from pandera.typing import DataFrame

from autotrader.core.schemas import StockPriceData


class _BaseComponent(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def _calculate(
        self, df: DataFrame[StockPriceData]
    ) -> pd.DataFrame | pd.Series: ...


class Feature(_BaseComponent):
    @abstractmethod
    def _calculate(self, df: DataFrame[StockPriceData]) -> pd.DataFrame: ...

    def __call__(self, df: DataFrame[StockPriceData]) -> pd.DataFrame:
        return self._calculate(df)


class Label(_BaseComponent):
    @abstractmethod
    def _calculate(self, df: DataFrame[StockPriceData]) -> pd.Series: ...

    def __call__(self, df: DataFrame[StockPriceData]) -> pd.Series:
        return self._calculate(df)
