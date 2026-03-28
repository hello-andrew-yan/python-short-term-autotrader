from autotrader.core.stock import StockHistory
from autotrader.extension.malyarovich import ForwardReturn, MalyarovichSMA


def main() -> None:
    df = StockHistory(["AAPL", "NVDA"]).get_data(verbose=True)

    features = MalyarovichSMA()(df)
    labels = ForwardReturn(gain_threshold=0.02)(df)

    combined = features.join(labels, how="inner")
    X = combined[features.columns]
    y = combined[labels.name]

    print(X)
    print(y)


if __name__ == "__main__":
    main()
