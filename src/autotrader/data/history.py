from dataclasses import dataclass, field
from typing import cast

import pandas as pd
import pandera.pandas as pa
import yfinance as yf
from pandera.typing import DataFrame

from autotrader import logger
from autotrader.core.enum import Frequency
from autotrader.core.schema import HistoryFrame as F


@dataclass
class StockHistory:
    tickers: str | list[str]
    freq: Frequency = Frequency.WEEKLY
    _cache: DataFrame[F] | None = field(default=None, init=False, repr=False)

    @pa.check_types
    def _resample(self, df: pd.DataFrame) -> DataFrame[F]:
        result = (
            cast(pd.DataFrame, df.stack(level=0))  # noqa: PD013
            .groupby(level=F.Ticker)
            .resample(self.freq, level=F.Date)
            .agg(
                {
                    F.Open: "first",
                    F.High: "max",
                    F.Low: "min",
                    F.Close: "last",
                    F.Volume: "sum",
                }
            )
            .dropna()
            .sort_index()
        )
        return cast(DataFrame[F], result)

    def __call__(
        self, verbose: bool = False, refresh: bool = False
    ) -> DataFrame[F]:
        if self._cache is not None and not refresh:
            logger.info("Returning cached data for %s", self.tickers)
            return self._cache

        tickers = (
            [self.tickers] if isinstance(self.tickers, str) else self.tickers
        )

        df = yf.download(
            tickers,
            period="max",
            group_by="ticker",
            auto_adjust=True,
            progress=verbose,
        )

        if df is None or df.empty:
            logger.warning("No data returned for %s", self.tickers)
            return cast(DataFrame[F], pd.DataFrame())

        self._cache = self._resample(df)
        logger.info("Downloaded %d rows for %s", len(self._cache), self.tickers)
        return self._cache
