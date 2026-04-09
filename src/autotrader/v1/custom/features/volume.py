import pandas as pd
from pandas_ta.statistics import zscore
from pandera.typing import DataFrame

from autotrader.v1.core.base import Feature
from autotrader.v1.core.schemas import StockPriceData as D


class VolumeZ(Feature):
    def __init__(self, period: int = 20):
        self.period = period

    def _calculate(self, df: DataFrame[D]) -> pd.DataFrame:
        v_z = df.groupby(level=D.Ticker)[D.Volume].transform(
            lambda x: zscore(x, length=self.period)
        )

        return pd.DataFrame(
            {f"Volume_Z_{self.period}": v_z}, index=df.index
        ).dropna()
