from typing import cast

import pandas as pd
import yfinance as yf

from autotrader import logger


class StockHistory:
    DEFAULT_FREQ = "W-FRI"
    ALLOWED_FREQS = (DEFAULT_FREQ, "M", "Q", "Y")

    def __init__(self, tickers: str | list[str], freq: str = DEFAULT_FREQ):
        self.tickers = tickers
        self.freq = freq

        if self.freq not in self.ALLOWED_FREQS:
            raise ValueError(
                f"Invalid freq frequency '{self.freq}'. "
                f"Allowed values: {self.ALLOWED_FREQS}"
            )

    def get_data(self, verbose: bool = False) -> pd.DataFrame:
        logger.info("Downloading stock history for %s", self.tickers)

        df = yf.download(
            self.tickers,
            period="max",
            group_by="ticker",
            auto_adjust=True,
            progress=verbose,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        return self._resample(df)

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            # stack returns DataFrame | Series, cast asserts DataFrame
            cast(pd.DataFrame, df.stack(level=0))  # noqa: PD013 - melt is not equivalent here
            .groupby(level=1)
            .resample(self.freq, level=0)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
            .sort_index()
        )
