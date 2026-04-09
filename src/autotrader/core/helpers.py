import hashlib

import pandas as pd
from pandera.typing import DataFrame
from rich import box
from rich.console import Console
from rich.table import Table

from autotrader.core.schemas import PerformanceMetrics as PM


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()


def print_metrics(
    df: DataFrame[PM],
    min_precision: float | None = None,
    sort_by: str | list[str] | None = None,
    ascending: bool | list[bool] = False,
):
    console = Console()
    title = "Performance Metrics"
    if min_precision is not None:
        title += f", [yellow]min_precision={min_precision:.2f}[/yellow]"

    table = Table(title=title, box=box.SIMPLE, header_style="bold cyan")

    display_df = (
        df.sort_values(by=sort_by, ascending=ascending) if sort_by else df
    ).copy()

    prec_col = str(PM.Precision)
    display_df[prec_col] = [
        f"[{style}]{p:.4f}[/]"
        for p in display_df[prec_col]
        if (
            style := "bold green"
            if min_precision and p >= min_precision
            else "red"
        )
    ]

    display_df.index = [f"[bold]{i}[/bold]" for i in display_df.index]
    display_df = display_df.reset_index(names=str(PM.Ticker)).astype(str)

    for col in display_df.columns:
        table.add_column(
            col, justify="left" if col == str(PM.Ticker) else "right"
        )

    for row in display_df.to_numpy():
        table.add_row(*row)

    console.print(table)
