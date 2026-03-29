import logging

from autotrader.data.history import StockHistory

FEATURES = ["NVDA"]
HELPERS = ["SPY"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="(%(name)s) %(message)s")

    history = StockHistory(tickers=[*FEATURES, *HELPERS])
    df = history(verbose=True)

    print(df)


if __name__ == "__main__":
    main()
