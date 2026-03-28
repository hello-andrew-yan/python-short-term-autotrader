from autotrader.core import StockDatasetBuilder, StockHistory
from autotrader.extension.malyarovich import ForwardReturn, MalyarovichSMA


def main() -> None:
    history = StockHistory(["AAPL", "NVDA"])

    feature = MalyarovichSMA()
    label = ForwardReturn(gain_threshold=0.02)

    builder = StockDatasetBuilder(history, features=[feature], label=label)
    dataset = builder()

    print(dataset)
    print(dataset.X)
    print(dataset.y)


if __name__ == "__main__":
    main()
