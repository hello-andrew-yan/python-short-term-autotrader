import pandas as pd
from pandera.typing import DataFrame

from autotrader.core.base import Label
from autotrader.core.schemas import StockPriceData as D


class ForwardReturn(Label):
    def __init__(self, horizon: int = 1, gain_threshold: float = 0.01):
        self.horizon = horizon
        self.gain_threshold = gain_threshold

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
            .rename(f"Return_{self.horizon}_{self.gain_threshold}")
        )
