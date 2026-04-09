import pandas as pd
from pandera.typing import DataFrame

from autotrader.core.base import Label
from autotrader.core.schemas import StockPriceData as D


class WeekOpenReturn(Label):
    def __init__(self, horizon: int = 1, gain_threshold: float = 0.01):
        self.horizon = horizon
        self.gain_threshold = gain_threshold

    def _calculate(self, df: DataFrame[D]) -> pd.Series:
        entry_price = df.groupby(level=D.Ticker)[D.Open].shift(-1)
        exit_price = df.groupby(level=D.Ticker)[D.Open].shift(-2)

        returns = (exit_price / entry_price) - 1

        return (
            (returns >= self.gain_threshold)
            .astype(int)
            .loc[returns.notna()]
            .rename(f"Monday_Return{self.gain_threshold}")
        )
