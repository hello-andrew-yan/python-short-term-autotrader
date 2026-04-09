import pandas as pd
from pandas_ta.overlap import sma
from pandera.typing import DataFrame

from autotrader.core.base import Feature
from autotrader.core.schemas import StockPriceData as D


class _SMABase(Feature):
    def _get_sma(self, df: DataFrame[D], period: int) -> pd.Series:
        return df.groupby(level=D.Ticker)[D.Close].transform(
            lambda x: sma(x, length=period)
        )


class SMANorm(_SMABase):
    def __init__(self, period: int = 20):
        self.period = period

    def _calculate(self, df: DataFrame[D]) -> pd.DataFrame:
        s_sma = self._get_sma(df, self.period)
        norm = (df[D.Close] - s_sma) / s_sma

        return pd.DataFrame(
            {f"SMA_Norm_{self.period}": norm}, index=df.index
        ).dropna()


class SMASlope(_SMABase):
    def __init__(self, period: int = 20, lookback: int = 5):
        self.period = period
        self.lookback = lookback

    def _calculate(self, df: DataFrame[D]) -> pd.DataFrame:
        sma_series = self._get_sma(df, self.period)
        slope = sma_series.groupby(level=D.Ticker).pct_change(
            periods=self.lookback
        )

        return pd.DataFrame(
            {f"SMA_Slope_{self.period}_{self.lookback}": slope}, index=df.index
        ).dropna()


class SMASpread(_SMABase):
    def __init__(self, short: int = 20, long: int = 200):
        self.short = short
        self.long = long

    def _calculate(self, df: DataFrame[D]) -> pd.DataFrame:
        s_sma = self._get_sma(df, self.short)
        l_sma = self._get_sma(df, self.long)
        spread = (s_sma - l_sma) / l_sma * 100

        return pd.DataFrame(
            {f"SMA_Spread_{self.short}_{self.long}": spread}, index=df.index
        ).dropna()
