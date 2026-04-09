from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index, Series


class StockPriceIndex(pa.DataFrameModel):
    Ticker: Index[str]
    Date: Index[pa.DateTime]


class StockPriceData(StockPriceIndex):
    Open: Series[float]
    High: Series[float]
    Low: Series[float]
    Close: Series[float]
    Volume: Series[float] = pa.Field(coerce=True)


class TradeResult(pa.DataFrameModel):
    Total: Series[int]
    Wins: Series[int]
    Losses: Series[int]
    Precision: Series[float]


@dataclass(frozen=True)
class DateWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_string(cls, start: str, end: str) -> "DateWindow":
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)

        if s >= e:
            raise ValueError(f"Start {s} must be strictly before end {e}")

        return cls(s, e)

    def extend(self, other: "DateWindow") -> "DateWindow":
        if not isinstance(other, DateWindow):
            raise TypeError("Can only extend with another DateWindow")

        return DateWindow(
            start=min(self.start, other.start), end=max(self.end, other.end)
        )
