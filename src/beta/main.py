import logging
from itertools import chain
from typing import cast

from rich import print as rprint
from yahooquery import Ticker

from autotrader import beta
from autotrader.core.schemas import DateWindow
from autotrader.stock.builder import DatasetBuilder
from autotrader.stock.history import StockHistory
from beta.core.helpers import print_results
from beta.extension.custom import SMA, ForwardReturn
from beta.model.dataset import DatasetSplit
from beta.model.predictor import PredictorConfig, StockPredictor
from beta.search.greedy import GreedySearch


@beta
def _get_all_holdings(symbols: list[str]) -> list[str]:
    tickers = Ticker(symbols)
    data = cast(dict, tickers.fund_holding_info)

    all_symbols = []
    for symbol in symbols:
        holdings = data.get(symbol, {}).get("holdings", [])
        all_symbols.extend([h["symbol"] for h in holdings if "symbol" in h])

    return list(set(all_symbols))


ETFS = [
    "IGV",  # iShares Expanded Tech-Software Sector ETF
    "SMH",  # VanEck Semiconductor ETF
    "SOXX",  # iShares Semiconductor ETF
    "QQQ",  # Invesco QQQ Trust, Series 1
    "XSD",  # State Street SPDR S&P Semiconductor ETF
]

PROMISING = [
    "ANET",  # Arista Networks, Inc.
    "APP",  # AppLovin Corporation
    "ARM",  # Arm Holdings plc
    "CLS",  # Celestica Inc.
    "LSCC",  # Lattice Semiconductor Corporation
    "MPWR",  # Monolithic Power Systems, Inc.
    "PLTR",  # Palantir Technologies Inc.
    "WOLF",  # Wolfspeed, Inc.
]
ANCHOR_FEATURES = ["WDC"]
ANCHOR_HELPERS = []

UNORGANISED = []
IGNORE = []

CANDIDATES = list(
    set(
        chain(
            ETFS,
            _get_all_holdings(ETFS),
            PROMISING,
            ANCHOR_FEATURES,
            ANCHOR_HELPERS,
        )
    )
    - set(IGNORE)
)

# May reduce the parameters slightly to increase search speed.
SEARCH_CONFIG = PredictorConfig()

DETERMINISTIC_CONFIG = PredictorConfig(
    nthread=1,
    tree_method="exact",
    device="cpu",
    random_state=42,
)

MIN_PRECISION = 0.65


if __name__ == "__main__":
    builder = DatasetBuilder(
        history=StockHistory(
            CANDIDATES,
            DateWindow.from_string("1980-01-01", "2026-12-31"),
        ),
        features=SMA(),
        label=ForwardReturn(gain_threshold=0.015),
    )
    dataset = builder.build()
    split = DatasetSplit.from_dates(
        dataset=dataset,
        train=DateWindow.from_string("1980-01-01", "2022-12-31"),
        val=DateWindow.from_string("2023-01-01", "2023-12-31"),
        test=DateWindow.from_string("2024-01-01", "2024-12-31"),
    )

    search = GreedySearch(
        split=split,
        pred_config=SEARCH_CONFIG,
        candidates=CANDIDATES,
        anchor_features=ANCHOR_FEATURES,
        anchor_helpers=ANCHOR_HELPERS,
        min_precision=MIN_PRECISION,
        min_trades=15,
        diversity_weight=0.5,
        score_delta=0.05,
        max_patience=2,
    )

    features, helpers, zeros = search.run()
    rprint(f"Features: {features}")
    rprint(f"Helpers: {helpers}")
    rprint(f"Zeros: {zeros}")

    if search.split is None:
        raise RuntimeError("Search split is unexpectedly None")

    logging.getLogger().setLevel(logging.INFO)

    split = search.split.get_tickers([*features, *helpers])
    predictor = StockPredictor(config=DETERMINISTIC_CONFIG)
    predictor.train(split.train, split.val, verbose=False)

    result = predictor.test(
        dataset.between(
            DateWindow.from_string("2025-01-01", "2025-12-31")
        ).tickers([*features, *helpers])
    )
    print_results(result, min_precision=MIN_PRECISION)

    result = predictor.test(
        dataset.between(
            DateWindow.from_string("2026-01-01", "2026-12-31")
        ).tickers([*features, *helpers])
    )
    print_results(result, min_precision=MIN_PRECISION)
