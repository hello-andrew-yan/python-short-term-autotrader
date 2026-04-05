import hashlib

import pandas as pd


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()
