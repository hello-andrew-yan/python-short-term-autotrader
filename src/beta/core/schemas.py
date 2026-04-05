import pandera.pandas as pa
from pandera.typing import Series


class TradeResult(pa.DataFrameModel):
    Total: Series[int]
    Wins: Series[int]
    Losses: Series[int]
    Precision: Series[float]
