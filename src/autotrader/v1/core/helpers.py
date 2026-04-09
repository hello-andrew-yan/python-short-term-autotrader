import hashlib

import pandas as pd
from pandera.typing import DataFrame
from rich.console import Console
from rich.table import Table

from autotrader.v1.core.schemas import TradeResult


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()


def print_results(
    df: DataFrame[TradeResult],
    min_precision: float | None = None,
):
    console = Console()
    table = Table(header_style="bold magenta")
    table.add_column("Ticker", style="dim")

    for col in df.columns:
        table.add_column(str(col))

    for idx, row in df.iterrows():
        values = [str(idx)]
        precision = row.get("Precision") if "Precision" in df.columns else None
        highlight_row = (
            min_precision is not None
            and precision is not None
            and precision < min_precision
        )

        for col, val in row.items():
            if col == "Precision" and isinstance(val, (int, float)):
                formatted = f"{val:.4f}"
                if min_precision is not None:
                    color = "green" if val >= min_precision else "red"
                    values.append(f"[{color}]{formatted}[/{color}]")
                else:
                    values.append(formatted)
            else:
                values.append(str(val))

        if highlight_row:
            values[0] = f"[red]{values[0]}[/red]"

        table.add_row(*values)

    console.print(table)
