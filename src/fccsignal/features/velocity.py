"""Filing velocity and acceleration, computed PIT-safe.

Leak trap this module exists to avoid: a rolling z-score whose mean/std
window *includes the current observation* is mildly self-referential; a
z-score computed over the full sample is catastrophically leaky. Here
every baseline statistic is computed on strictly-prior data via
`shift(1)` before the rolling window is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_counts(
    events: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "grant_date",
) -> pd.DataFrame:
    """(ticker, date) -> n_filings, on a dense daily calendar per ticker
    spanning that ticker's first-to-last event date."""
    ev = events.copy()
    ev[date_col] = pd.to_datetime(ev[date_col]).dt.normalize()
    out = []
    for tkr, grp in ev.groupby(ticker_col):
        counts = grp.groupby(date_col).size()
        idx = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
        dense = counts.reindex(idx, fill_value=0)
        out.append(
            pd.DataFrame(
                {ticker_col: tkr, "date": idx, "n_filings": dense.values}
            )
        )
    return pd.concat(out, ignore_index=True)


def velocity_features(
    counts: pd.DataFrame,
    short_window: int = 90,
    long_window: int = 365,
    min_history: int = 180,
) -> pd.DataFrame:
    """Trailing sums plus an acceleration z-score.

    accel_z at date T:
        x   = trailing `short_window`-day filing count ending at T
        mu  = mean of x over the prior `long_window` days, EXCLUDING T
        sd  = std of the same strictly-prior window
        z   = (x - mu) / sd          (NaN until `min_history` days exist)
    """
    frames = []
    for tkr, grp in counts.groupby("ticker"):
        g = grp.sort_values("date").set_index("date")
        x = g["n_filings"].rolling(short_window, min_periods=1).sum()
        prior = x.shift(1)  # strictly-prior baseline
        mu = prior.rolling(long_window, min_periods=min_history).mean()
        sd = prior.rolling(long_window, min_periods=min_history).std()
        z = (x - mu) / sd.replace(0.0, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": tkr,
                    "date": g.index,
                    "trailing_filings": x.values,
                    "accel_z": z.values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
