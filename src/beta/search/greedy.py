from dataclasses import dataclass

from pandera.typing import DataFrame
from tqdm import tqdm

from autotrader import beta
from beta.core.schemas import TradeResult as T
from beta.model.dataset import DatasetSplit
from beta.model.predictor import PredictorConfig, StockPredictor


@beta
@dataclass
class GreedySearch:
    split: DatasetSplit
    pred_config: PredictorConfig

    candidates: list[str]
    anchor_features: list[str]
    anchor_helpers: list[str]

    min_precision: float
    min_trades: int
    diversity_weight: float

    score_delta: float
    max_patience: int

    def _evaluate(self, tickers: list[str]) -> DataFrame[T] | None:
        data = self.split.get_tickers(tickers)
        if data.train.X.empty or data.val.X.empty or data.test.X.empty:
            return None

        return (
            StockPredictor(config=self.pred_config)
            .train(data.train, data.val, verbose=False)
            .test(data.test)
        )

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
        feature_ratio = len(feature_tickers) / max(len(valid), 1)

        coverage = (vol_ratio * ticker_ratio) * (1 + feature_ratio) / 2
        d = self.diversity_weight
        return precision * (1 - d) + coverage * d

    def _run_pass(  # noqa: PLR0913
        self,
        active: list[str],
        remaining: list[str],
        zero_counts: dict[str, int],
        pass_idx: int,
        patience: int,
        best_score: float,
        last_added: str | None,
    ) -> tuple[str | None, float, DataFrame[T] | None]:
        best_candidate = None
        new_score = 0.0
        best_trades = None

        with tqdm(
            remaining,
            leave=False,
            colour="green",
            bar_format="{l_bar}{bar}[{elapsed}<{remaining}] {postfix}",
        ) as pbar:
            pbar.set_description(
                f"Pass {pass_idx} | best={best_score:.4f}, "
                f"added={last_added or 'N/A'}"
            )
            for c in pbar:
                pbar.set_postfix(
                    candidate=c,
                    patience=f"{patience}/{self.max_patience}",
                    best=f"{new_score:.4f}",
                )

                result = self._evaluate(active + [c])
                if result is None:
                    continue

                score = self._score(result)
                if score == 0.0:
                    zero_counts[c] = zero_counts.get(c, 0) + 1
                    continue

                if score > new_score:
                    new_score = score
                    best_candidate = c
                    best_trades = result

        return best_candidate, new_score, best_trades

    def _search(self) -> tuple[list[str], list[str], list[tuple[str, int]]]:
        active = list(set(self.anchor_features + self.anchor_helpers))
        remaining = [c for c in self.candidates if c not in active]

        best_trades = self._evaluate(active)
        best_score = (
            self._score(best_trades) if best_trades is not None else 0.0
        )
        patience = 0
        pass_idx = 0
        last_added = None
        zero_counts: dict[str, int] = {}

        while remaining:
            pass_idx += 1
            best_candidate, new_score, pass_trades = self._run_pass(
                active,
                remaining,
                zero_counts,
                pass_idx,
                patience,
                best_score,
                last_added,
            )

            if best_candidate is None or (
                best_score == 0.0 and new_score == 0.0
            ):
                break

            if new_score >= best_score:
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
            active.sort()

            remaining.remove(best_candidate)
            last_added = best_candidate

            if pass_trades is not None:
                best_trades = pass_trades

        if best_trades is None:
            raise ValueError("No trades data available for final evaluation.")

        mask = (best_trades[T.Precision] >= self.min_precision) & (
            best_trades[T.Total] >= self.min_trades
        )
        passing = set(best_trades[mask].index.tolist())
        features = list(
            (passing - set(self.anchor_helpers)) | set(self.anchor_features)
        )
        helpers = [t for t in active if t not in features]
        zeros = sorted(
            [(c, zero_counts.get(c, 0)) for c in remaining],
            key=lambda x: x[1],
            reverse=True,
        )
        return features, helpers, zeros

    def run(self) -> tuple[list[str], list[str], list[tuple[str, int]]]:
        return self._search()
