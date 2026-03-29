from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from autotrader.core.schema import HistoryFrame as F


@dataclass
class _Base(ABC):
    @abstractmethod
    def _calculate(self, df: DataFrame[F]) -> DataFrame | Series: ...

    @pa.check_types
    def __call__(self, df: DataFrame[F]) -> DataFrame | Series:
        return self._calculate(df)


class Feature(_Base):
    @abstractmethod
    def _calculate(self, df: DataFrame[F]) -> DataFrame: ...


class Label(_Base):
    @abstractmethod
    def _calculate(self, df: DataFrame[F]) -> Series: ...

    @pa.check_types
    def __call__(self, df: DataFrame[F]) -> Series:
        return self._calculate(df)
