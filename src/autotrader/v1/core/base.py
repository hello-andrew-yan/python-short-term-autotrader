from abc import ABC, abstractmethod

import pandas as pd
from pandera.typing import DataFrame

from autotrader.v1.core.schemas import StockPriceData


class Feature(ABC):
    @abstractmethod
    def _calculate(self, df: DataFrame[StockPriceData]) -> pd.DataFrame: ...

    def __call__(self, df: DataFrame[StockPriceData]) -> pd.DataFrame:
        return self._calculate(df)


class Label(ABC):
    @abstractmethod
    def _calculate(self, df: DataFrame[StockPriceData]) -> pd.Series: ...

    def __call__(self, df: DataFrame[StockPriceData]) -> pd.Series:
        return self._calculate(df)
