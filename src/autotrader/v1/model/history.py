from dataclasses import dataclass
from functools import cached_property
from typing import cast

import pandas as pd
import yfinance as yf
from pandera.typing import DataFrame

from autotrader.v1.core.schemas import DateWindow, StockPriceData


@dataclass(frozen=True)
class StockHistory:
    tickers: str | list[str]
    window: DateWindow

    @cached_property
    def _raw(self) -> pd.DataFrame:
        tickers_list = (
            [self.tickers] if isinstance(self.tickers, str) else self.tickers
        )
        raw = yf.download(
            tickers_list,
            start=self.window.start,
            end=self.window.end,
            auto_adjust=False,
            progress=False,
            period=None,
            interval="1d",
        )

        if raw is None or raw.empty:
            raise ValueError(f"No data returned for {self.tickers}")

        return raw

    def get_data(self, freq: str | None = None) -> DataFrame[StockPriceData]:
        if freq is None:
            freq = "W-FRI"

        return StockPriceData.validate(
            cast(
                pd.DataFrame,
                self._raw.stack(level="Ticker", future_stack=True),  # noqa: PD013
            )
            .groupby(level="Ticker")
            .resample(freq, level="Date")
            .agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
                Volume=("Volume", "sum"),
            )
            .dropna()
        )
