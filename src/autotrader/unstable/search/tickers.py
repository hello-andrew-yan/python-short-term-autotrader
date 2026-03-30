from itertools import chain
from typing import cast

from yahooquery import Ticker

from autotrader import beta


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
    "ADI",  # Analog Devices, Inc.
    "AMAT",  # Applied Materials, Inc.
    "AMD",  # Advanced Micro Devices, Inc.
    "AMZN",  # Amazon.com, Inc.
    "GOOG",  # Alphabet Inc.
    "KLAC",  # KLA Corporation
    "LRCX",  # Lam Research Corporation
    "TSLA",  # Tesla, Inc.
    "TSM",  # Taiwan Semiconductor Manufacturing Company Limited
    "TXN",  # Texas Instruments Incorporated
    "WMT",  # Walmart Inc.
]

UNORGANISED = [
    "AMKR",  # Amkor Technology, Inc.
    "ANET",  # Arista Networks, Inc.
    "ARM",  # Arm Holdings plc
    "AZTA",  # Azenta, Inc.
    "COHR",  # Coherent Corp.
    "ENTG",  # Entegris, Inc.
    "FORM",  # FormFactor, Inc.
    "IPGP",  # IPG Photonics Corporation
    "LSCC",  # Lattice Semiconductor Corporation
    "MCHP",  # Microchip Technology Incorporated
    "MDRX",  # Veradigm Inc.
    "MKSI",  # MKS Instruments, Inc.
    "MPWR",  # Monolithic Power Systems, Inc.
    "MRVL",  # Marvell Technology, Inc.
    "ON",  # ON Semiconductor Corporation
    "ONTO",  # Onto Innovation Inc.
    "SMCI",  # Super Micro Computer, Inc.
    "STX",  # Seagate Technology Holdings plc
    "TEL",  # TE Connectivity Ltd.
    "VRT",  # Vertiv Holdings Co
]
IGNORE = []

CANDIDATES = list(
    set(chain(_get_all_holdings(ETFS), PROMISING, UNORGANISED)) - set(IGNORE)
)
