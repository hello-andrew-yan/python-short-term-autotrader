import logging
from dataclasses import dataclass

from pandera.typing import DataFrame
from tqdm import tqdm

from autotrader import beta
from autotrader.core.base import Feature, Label
from autotrader.core.schema import DateWindow
from autotrader.core.schema import TradeResult as T
from autotrader.data.history import StockHistory
from autotrader.extension.custom import SMA, ForwardReturn
from autotrader.ml import DatasetBuilder, DatasetSplit, StockPredictor
from autotrader.unstable.tickers import CANDIDATES


@dataclass
class SearchDataConfig:
    features: Feature | list[Feature]
    label: Label
    train: DateWindow
    val: DateWindow
    test: DateWindow


@beta
@dataclass
class GreedySearch:
    candidates: list[str]
    anchor_features: list[str]
    anchor_helpers: list[str]
    data: SearchDataConfig
    min_precision: float
    min_trades: int
    diversity_weight: float
    score_delta: float
    max_patience: int
    split: DatasetSplit | None = None

    def _build_split(self) -> None:
        if self.split is not None:
            return

        all_tickers = list(
            set(self.candidates + self.anchor_features + self.anchor_helpers)
        )
        builder = DatasetBuilder(
            StockHistory(tickers=all_tickers),
            self.data.features,
            self.data.label,
        )
        self.split = DatasetSplit.from_dates(
            builder.build(),
            train=self.data.train,
            val=self.data.val,
            test=self.data.test,
        )

    def _evaluate(self, tickers: list[str]) -> DataFrame[T]:
        if self.split is None:
            raise ValueError("Split must be built before evaluating.")

        data = self.split.get_tickers(tickers)
        return StockPredictor().train(data.train, data.val).test(data.test)

    def _score(self, trades: DataFrame[T]) -> float:
        valid = trades[
            (trades[T.Precision] >= self.min_precision)
            & (trades[T.Total] >= self.min_trades)
        ]
        if valid.empty:
            return 0.0

        total = valid[T.Total].sum()
        precision = (valid[T.Precision] * valid[T.Total]).sum() / total
        vol_ratio = total / trades[T.Total].sum()
        ticker_ratio = len(valid) / len(trades)

        feature_tickers = [
            t for t in valid.index if t not in self.anchor_helpers
        ]
        feature_ratio = len(feature_tickers) / max(len(self.candidates), 1)

        coverage = (vol_ratio * ticker_ratio) * (1 + feature_ratio) / 2
        d = self.diversity_weight
        return precision * (1 - d) + coverage * d

    def _search(self) -> tuple[list[str], list[str]]:
        active = list(set(self.anchor_features + self.anchor_helpers))
        remaining = [c for c in self.candidates if c not in active]

        best_trades = self._evaluate(active)
        best_score = self._score(best_trades)
        patience = 0
        pass_idx = 0

        while remaining:
            pass_idx += 1
            scores = {}

            with tqdm(
                remaining,
                leave=False,
                colour="green",
                bar_format="{l_bar}{bar} [{elapsed}] {postfix}",
            ) as pbar:
                pbar.set_description(f"Pass {pass_idx} | best={best_score:.4f}")
                for c in pbar:
                    scores[c] = self._score(self._evaluate(active + [c]))
                    pbar.set_postfix(
                        candidate=c,
                        patience=f"{patience}/{self.max_patience}",
                        score=f"{scores[c]:.4f}",
                    )

            best_candidate, new_score = max(scores.items(), key=lambda x: x[1])

            if best_score == 0.0 and new_score == 0.0:
                break
            elif new_score >= best_score:
                best_score = new_score
                patience = 0
            elif (
                new_score >= best_score - self.score_delta
                and patience < self.max_patience
            ):
                best_score = new_score
                patience += 1
            else:
                break

            active.append(best_candidate)
            remaining.remove(best_candidate)
            best_trades = self._evaluate(active)

        mask = (best_trades[T.Precision] >= self.min_precision) & (
            best_trades[T.Total] >= self.min_trades
        )
        features = [
            t
            for t in best_trades[mask].index.tolist()
            if t not in self.anchor_helpers
        ]
        helpers = [t for t in active if t not in features]
        return features, helpers

    def run(self) -> tuple[list[str], list[str]]:
        self._build_split()
        return self._search()


def main() -> None:
    data = SearchDataConfig(
        features=SMA(),
        label=ForwardReturn(gain_threshold=0.015),
        train=DateWindow("1980-01-01", "2023-02-27"),
        val=DateWindow("2023-02-28", "2025-02-28"),
        test=DateWindow("2025-03-01", "2026-03-01"),
    )
    searcher = GreedySearch(
        candidates=CANDIDATES,
        anchor_features=["MU", "WDC"],
        anchor_helpers=["SPY"],
        data=data,
        min_precision=0.65,
        min_trades=10,
        diversity_weight=0.5,
        score_delta=0.10,
        max_patience=2,
    )
    features, helpers = searcher.run()
    print(f"Features: {features}")
    print(f"Helpers: {helpers}")

    if searcher.split is None:
        raise RuntimeError("Split was not built.")

    logging.basicConfig(level=logging.INFO, format="(%(name)s) %(message)s")

    split = searcher.split.get_tickers([*features, *helpers])
    predictor = StockPredictor()
    predictor.train(split.train, split.val, verbose=True)
    predictor.test(split.test, ignore=helpers)


if __name__ == "__main__":
    main()
