from typing import cast

import pandas as pd
from pandas_ta.overlap import sma
from pandas_ta.statistics import zscore
from pandera.typing import DataFrame, Series

from autotrader.core.base import Feature, Label
from autotrader.core.schemas import StockPriceData as D


class SMA(Feature):
    def __init__(
        self,
        short: int = 20,
        long: int = 200,
        slope: int = 4,
        volume: int = 20,
    ):
        self.short = short
        self.long = long
        self.slope = slope
        self.volume = volume

    def _calculate(self, df: DataFrame[D]) -> DataFrame:
        group = df.groupby(level=D.Ticker)
        close = df[D.Close]
        s_sma = group[D.Close].transform(lambda x: sma(x, length=self.short))
        l_sma = group[D.Close].transform(lambda x: sma(x, length=self.long))

        result = pd.DataFrame(
            {
                f"SMA_Norm_{self.short}": (close - s_sma) / s_sma,
                f"SMA_Norm_{self.long}": (close - l_sma) / l_sma,
                f"SMA_Spread_{self.short}_{self.long}_PCT": (s_sma - l_sma)
                / l_sma
                * 100,
                f"SMA_{self.short}_Slope_{self.slope}": s_sma.groupby(
                    level=D.Ticker
                ).pct_change(periods=self.slope),
                f"Volume_Z_{self.short}": group[D.Volume].transform(
                    lambda x: zscore(x, length=self.volume)
                ),
            },
            index=df.index,
        )
        return cast(DataFrame, result.dropna())


class ForwardReturn(Label):
    def __init__(self, gain_threshold: float, horizon: int = 1):
        self.gain_threshold = gain_threshold
        self.horizon = horizon

    def _calculate(self, df: DataFrame[D]) -> Series:
        future_price = df.groupby(level=D.Ticker)[D.Close].shift(-self.horizon)
        forward_return = future_price / df[D.Close] - 1
        label = (forward_return >= self.gain_threshold).astype(int)
        label.name = (
            f"Forward_Return_{self.horizon}_Gain_PCT_{self.gain_threshold}"
        )

        result = label.where(future_price.notna())
        return cast(Series, result.dropna())
