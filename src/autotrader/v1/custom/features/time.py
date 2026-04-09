import calendar

import pandas as pd
from pandera.typing import DataFrame

from autotrader.v1.core.base import Feature
from autotrader.v1.core.schemas import StockPriceData as D


class MonthFeature(Feature):
    def __init__(self, focus_months: list[int] | None = None):
        self.focus_months = focus_months or list(range(1, 13))

    def _calculate(self, df: DataFrame[D]) -> pd.DataFrame:
        times = pd.to_datetime(df.index.get_level_values(D.Date))
        flags = {
            f"Is_{calendar.month_name[m]}": (times.month == m).astype(int)
            for m in self.focus_months
        }
        return pd.DataFrame(flags, index=df.index)
