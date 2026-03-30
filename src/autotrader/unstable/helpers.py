import hashlib

import pandas as pd


def get_fingerprint(df: pd.DataFrame, include_index: bool = True) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=include_index)

    hash_array = row_hashes.to_numpy(dtype="uint64", copy=False)

    return hashlib.sha256(hash_array.tobytes()).hexdigest()
