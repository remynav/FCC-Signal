"""Market data puller (skeleton).

Needs:
- Daily total returns for the hardware universe INCLUDING delisted and
  acquired names, with delisting returns applied. In a small/mid-cap
  hardware universe, acquisition is a common POSITIVE outcome; dropping
  acquired names biases against the signal (inverted survivorship).
- Fama-French daily factors (Ken French data library).
- ADV (average daily dollar volume) for the capacity analysis.

Free-tier reality check: fully delisting-adjusted history is the one
place free sources are weakest. Options, in order of preference:
survivorship-aware vendor trial, WRDS/CRSP via university access
(USC has WRDS — use it), or documented best-effort from free sources
with the limitation stated in the writeup.
"""

from __future__ import annotations

import pandas as pd

BUSINESS_COLS_PRICES = ["date", "ticker", "ret", "close", "volume", "delist_ret"]
BUSINESS_COLS_FACTORS = ["date", "mkt_rf", "smb", "hml", "rf"]


def pull_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:  # pragma: no cover
    raise NotImplementedError


def pull_ff_factors(start: str, end: str) -> pd.DataFrame:  # pragma: no cover
    raise NotImplementedError
