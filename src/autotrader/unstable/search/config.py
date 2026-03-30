from dataclasses import dataclass

from autotrader.core.base import Feature, Label
from autotrader.core.schema import (
    DateWindow,
)
from autotrader.extension.custom import SMA, ForwardReturn
from autotrader.ml.predictor import PredictorConfig


@dataclass
class DatasetConfig:
    features: Feature | list[Feature]
    label: Label
    train: DateWindow
    val: DateWindow
    test: DateWindow


DATASET_CONFIG = DatasetConfig(
    features=SMA(),
    label=ForwardReturn(gain_threshold=0.015),
    train=DateWindow("1980-01-01", "2021-12-31"),
    val=DateWindow("2022-01-01", "2024-12-31"),
    test=DateWindow("2025-01-01", "2026-03-01"),
)

# May reduce the parameters slightly to increase search speed.
SEARCH_CONFIG = PredictorConfig()

DETERMINISTIC_CONFIG = PredictorConfig(
    nthread=1,
    tree_method="exact",
    device="cpu",
    random_state=42,
)
