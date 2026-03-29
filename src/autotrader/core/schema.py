import pandera.pandas as pa
from pandera.typing import Index, Series


class StockHistoryFrame(pa.DataFrameModel):
    Ticker: Index[str]
    Date: Index[str]
    Open: Series[float]
    High: Series[float]
    Low: Series[float]
    Close: Series[float]
    Volume: Series[float]

    class Config:
        multiindex_coerce = True
