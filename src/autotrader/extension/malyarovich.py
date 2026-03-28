from collections.abc import Sequence

import pandas as pd
from pandas_ta.overlap import sma
from pandas_ta.statistics import zscore

from autotrader.core.base import Feature, Label


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

    @property
    def required_columns(self) -> Sequence[str]:
        return [self.CLOSE, self.VOLUME]

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        grouped = df.groupby(level=self.TICKER)
        t_close = grouped[self.CLOSE]
        t_volume = grouped[self.VOLUME]

        s_sma = t_close.transform(lambda x: sma(x, length=self.short))
        l_sma = t_close.transform(lambda x: sma(x, length=self.long))

        output[f"SMA_Norm_{self.short}"] = (df[self.CLOSE] - s_sma) / s_sma
        output[f"SMA_Norm_{self.long}"] = (df[self.CLOSE] - l_sma) / l_sma
        output[f"SMA_Spread_PCT_{self.short}_{self.long}"] = (
            (s_sma - l_sma) / l_sma * 100
        )
        # s_sma loses grouping post-transform, reapplied to maintain grouping
        output[f"SMA_{self.short}_Slope_{self.slope}"] = s_sma.groupby(
            level=self.TICKER
        ).pct_change(periods=self.slope)
        output[f"Volume_Z_{self.short}"] = t_volume.transform(
            lambda x: zscore(x, length=self.volume)
        )

        return output.dropna()


class ForwardReturn(Label):
    def __init__(self, gain_threshold: float, horizon: int = 1):
        self.gain_threshold = gain_threshold
        self.horizon = horizon

    @property
    def required_columns(self) -> Sequence[str]:
        return [self.CLOSE]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        future_price = df.groupby(level=self.TICKER)[self.CLOSE].shift(
            -self.horizon
        )
        forward_return = future_price / df[self.CLOSE] - 1
        label = (forward_return >= self.gain_threshold).astype(int)
        label.name = (
            f"Forward_Return_{self.horizon}_Gain_PCT_{self.gain_threshold}"
        )

        # Masks NaN rows introduced by shift
        return label[future_price.notna()]
