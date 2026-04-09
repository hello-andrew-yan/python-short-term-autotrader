import pandera.pandas as pa
from pandera.typing import Index, Series


class _BaseConfig(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = False
        ordered = True


class StockPriceIndex(_BaseConfig):
    Ticker: Index[str]
    Date: Index[pa.DateTime]


class StockPriceData(StockPriceIndex):
    Open: Series[float]
    High: Series[float]
    Low: Series[float]
    Close: Series[float]
    Volume: Series[float]


class PerformanceMetrics(_BaseConfig):
    Ticker: Index[str]
    Total: Series[int]
    Wins: Series[int]
    Losses: Series[int]
    Precision: Series[float]
