import math
from dataclasses import dataclass

from tqdm import tqdm

from autotrader import beta
from autotrader.extension.malyarovich import ForwardReturn, MalyarovichSMA
from autotrader.unstable.search.consts import CANDIDATES
from autotrader.unstable.search.helpers import (
    DataConfig,
    create_split,
    evaluate,
)


@dataclass
class SearchConfig:
    candidates: list[str]
    anchor_features: list[str]
    anchor_helpers: list[str]

    min_trades: int = 10
    min_precision: float = 0.65

    max_patience: int = 2
    tolerance: float = 0.10

    diversity_weight: float = 1.0


def _score(
    scores: dict[str, float],
    trades: dict[str, int],
    cfg: SearchConfig,
) -> float:
    valid = [
        t
        for t in scores
        if scores[t] >= cfg.min_precision and trades[t] >= cfg.min_trades
    ]
    if not valid:
        return 0.0

    total_valid_trades = sum(trades[t] for t in valid)
    weighted_precision = (
        sum(scores[t] * trades[t] for t in valid) / total_valid_trades
    )

    div_component = math.log1p(len(valid)) / math.log1p(len(scores))
    vol_ratio = total_valid_trades / sum(trades.values())

    return (
        weighted_precision
        * (vol_ratio**0.5)
        * (div_component**cfg.diversity_weight)
    )


@beta
def greedy_search(cfg: DataConfig, s_cfg: SearchConfig):
    active = list(s_cfg.anchor_features + s_cfg.anchor_helpers)
    remaining = [c for c in s_cfg.candidates if c not in active]
    split = create_split(cfg, list(set(active + remaining)))

    best_score = _score(*evaluate(split, active, cfg.predictor_config), s_cfg)
    patience = 0
    pass_idx = 0
    while remaining:
        pass_idx += 1
        moves = {}
        with tqdm(
            remaining,
            leave=False,
            colour="green",
            bar_format="{l_bar}{bar} [{elapsed}] {postfix}",
        ) as pbar:
            pbar.set_description(f"Pass {pass_idx}, best={best_score:.4f}")
            for c in pbar:
                score = _score(
                    *evaluate(split, active + [c], cfg.predictor_config), s_cfg
                )

                moves[c] = score
                pbar.set_postfix(
                    candidate=c,
                    patience=f"{patience}/{s_cfg.max_patience}",
                    score=f"{score:.4f}",
                )

        ticker, new_score = max(moves.items(), key=lambda x: x[1])
        if new_score >= best_score:
            patience = 0
        elif (
            new_score >= best_score - s_cfg.tolerance
            and patience < s_cfg.max_patience
        ):
            patience += 1
        else:
            break

        best_score = new_score
        active.append(ticker)
        remaining.remove(ticker)

    final_p, final_c = evaluate(split, active, cfg.predictor_config)
    features = [
        t
        for t in active
        if final_p.get(t, 0) >= s_cfg.min_precision
        and final_c.get(t, 0) >= s_cfg.min_trades
    ]
    helpers = [t for t in active if t not in features]

    print(f"\nFinal Score: {best_score:.2%}")
    print("Features:", sorted(features))
    print("Helpers:", sorted(helpers))

    return features, helpers


def main() -> None:
    cfg = DataConfig(
        features=[MalyarovichSMA()],
        label=ForwardReturn(gain_threshold=0.015),
        train_range=("1980-01-01", "2024-06-30"),
        val_range=("2024-07-01", "2025-06-30"),
        test_range=("2025-07-01", "2026-03-05"),
    )

    greedy_search(
        cfg=cfg,
        s_cfg=SearchConfig(
            candidates=CANDIDATES,
            anchor_features=["WDC"],
            anchor_helpers=["SPY"],
        ),
    )


if __name__ == "__main__":
    main()
