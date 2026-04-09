import pandas as pd
from pandera.typing import DataFrame

from autotrader.v1.core.base import Label
from autotrader.v1.core.schemas import StockPriceData as D


class ForwardReturn(Label):
    def __init__(self, gain_threshold: float, horizon: int = 1):
        self.gain_threshold = gain_threshold
        self.horizon = horizon

    def _calculate(self, df: DataFrame[D]) -> pd.Series:
        returns = (
            df.groupby(level=D.Ticker)[D.Close]
            .shift(-self.horizon)
            .div(df[D.Close])
            .sub(1)
        )

        return (
            (returns >= self.gain_threshold)
            .astype(int)
            .loc[returns.notna()]
            .rename(f"Return_{self.horizon}H_{self.gain_threshold}G")
        )
