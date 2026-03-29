from typing import cast

import pandas as pd
from pandas_ta.overlap import sma
from pandas_ta.statistics import zscore
from pandera.typing import DataFrame, Series

from autotrader.core.base import Feature, Label
from autotrader.core.schema import StockHistoryFrame as F


class MalyarovichSMA(Feature):
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

    def _calculate(self, df: DataFrame[F]) -> DataFrame:
        output = pd.DataFrame(index=df.index)
        grouped = df.groupby(level=F.Ticker)
        t_close = grouped[F.Close]
        t_volume = grouped[F.Volume]

        s_sma = t_close.transform(lambda x: sma(x, length=self.short))
        l_sma = t_close.transform(lambda x: sma(x, length=self.long))

        output[f"SMA_Norm_{self.short}"] = (df[F.Close] - s_sma) / s_sma
        output[f"SMA_Norm_{self.long}"] = (df[F.Close] - l_sma) / l_sma
        output[f"SMA_Spread_PCT_{self.short}_{self.long}"] = (
            (s_sma - l_sma) / l_sma * 100
        )
        output[f"SMA_{self.short}_Slope_{self.slope}"] = s_sma.groupby(
            level=F.Ticker
        ).pct_change(periods=self.slope)
        output[f"Volume_Z_{self.short}"] = t_volume.transform(
            lambda x: zscore(x, length=self.volume)
        )

        return cast(DataFrame, output.dropna())


class ForwardReturn(Label):
    def __init__(self, gain_threshold: float, horizon: int = 1):
        self.gain_threshold = gain_threshold
        self.horizon = horizon

    def _calculate(self, df: DataFrame[F]) -> Series:
        future_price = df.groupby(level=F.Ticker)[F.Close].shift(-self.horizon)
        forward_return = future_price / df[F.Close] - 1
        label = (forward_return >= self.gain_threshold).astype(int)
        label.name = (
            f"Forward_Return_{self.horizon}_Gain_PCT_{self.gain_threshold}"
        )

        return cast(Series, label[future_price.notna()])
