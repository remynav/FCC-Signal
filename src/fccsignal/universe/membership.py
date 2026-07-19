"""Universe construction (skeleton).

Definition (pre-registered; changing it later is a new spec):
- US-listed, primary equity listing
- GICS industries: technology hardware, storage & peripherals;
  electronic equipment, instruments & components; communications
  equipment; consumer electronics adjacents
- Market cap between $150M and $10B measured AS OF each membership
  date (never today's cap — that alone injects survivorship)
- Includes names later delisted or acquired

Output: long frame (date, ticker, in_universe) at monthly frequency.
"""

from __future__ import annotations

import pandas as pd


def build_membership(
    listings: pd.DataFrame, caps: pd.DataFrame,
    cap_min: float = 150e6, cap_max: float = 10e9,
) -> pd.DataFrame:  # pragma: no cover - needs real listings data
    raise NotImplementedError
