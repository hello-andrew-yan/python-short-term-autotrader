from collections.abc import Sequence
from dataclasses import dataclass

from autotrader.core import StockDatasetBuilder, StockHistory, StockPredictor
from autotrader.core.base import Feature, Label
from autotrader.core.dataset import DatasetSplit


@dataclass
class DataConfig:
    features: Sequence[Feature]
    label: Label
    train_range: tuple[str, str]
    val_range: tuple[str, str]
    test_range: tuple[str, str]
    predictor_config: dict | None = None


def create_split(cfg: DataConfig, tickers: list[str]):
    builder = StockDatasetBuilder(
        history=StockHistory(tickers),
        features=cfg.features,
        label=cfg.label,
    )
    return builder.split(
        train_range=cfg.train_range,
        val_range=cfg.val_range,
        test_range=cfg.test_range,
    )


def evaluate(
    split: DatasetSplit,
    tickers: list[str],
    config: dict | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    data = split.filter_by_ticker(tickers)
    model = StockPredictor(config=config)
    model.train(data.train, data.val)
    trades = model.test(data.test)

    return (
        {str(k): float(v) for k, v in trades["Precision"].items()},
        {str(k): int(v) for k, v in trades["Total"].items()},
    )
