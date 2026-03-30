import logging

from rich import print as rprint

# from autotrader.data.history import StockHistory
from autotrader.ml import StockPredictor

# from autotrader.unstable.helpers import get_fingerprint
from autotrader.unstable.search.config import (
    DATASET_CONFIG,
    DETERMINISTIC_CONFIG,
    SEARCH_CONFIG,
)
from autotrader.unstable.search.greedy import GreedySearch
from autotrader.unstable.search.tickers import CANDIDATES


def main() -> None:
    # history = StockHistory(
    #     tickers=CANDIDATES,
    #     start=DATASET_CONFIG.train.start,
    #     end=DATASET_CONFIG.test.end,
    # )

    # df = history()
    # hash_id = get_fingerprint(df)
    # print(f"DEBUG | History Hash: {hash_id}")

    # return

    logging.basicConfig(level=logging.WARNING, format="(%(name)s) %(message)s")
    search = GreedySearch(
        candidates=CANDIDATES,
        anchor_features=["MU", "WDC"],
        anchor_helpers=["SMH", "QQQ"],
        data_cfg=DATASET_CONFIG,
        pred_cfg=SEARCH_CONFIG,
        min_precision=0.65,
        min_trades=10,
        diversity_weight=0.5,
        score_delta=0.10,
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
    predictor.train(split.train, split.val, verbose=True)
    predictor.test(split.test, ignore=helpers)


if __name__ == "__main__":
    main()
