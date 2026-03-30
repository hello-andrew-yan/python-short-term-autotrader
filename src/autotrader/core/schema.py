from typing import NamedTuple

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index, Series


class HistoryIndex(pa.DataFrameModel):
    Ticker: Index[str]
    Date: Index[pa.DateTime]

    class Config:
        multiindex_coerce = True


class HistoryFrame(HistoryIndex):
    Open: Series[float]
    High: Series[float]
    Low: Series[float]
    Close: Series[float]
    Volume: Series[float]


class DateWindow(NamedTuple):
    start: str | pd.Timestamp
    end: str | pd.Timestamp


class TradeResult(pa.DataFrameModel):
    Total: Series[int]
    Wins: Series[int]
    Losses: Series[int]
    Precision: Series[float]
