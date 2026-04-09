from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    # TODO - May need to incorporate UTC conversion later down the line
    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                f"Start date ({self.start}) must be "
                f"strictly before end date ({self.end})"
            )

    @classmethod
    def from_string(cls, start: str, end: str) -> "DateWindow":
        return cls(pd.Timestamp(start), pd.Timestamp(end))

    def __str__(self) -> str:
        return f"{self.start.date()} to {self.end.date()}"
