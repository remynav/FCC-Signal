"""Research protocol utilities: walk-forward splits, Benjamini-Hochberg
control across the pre-registered spec grid, and placebo event dates.

The spec grid lives in configs/spec_grid.yaml and is committed BEFORE
any evaluation is run — that commit hash is the pre-registration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# -- walk-forward -------------------------------------------------------
@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp    # exclusive
    test_start: pd.Timestamp   # == train_end plus embargo
    test_end: pd.Timestamp     # exclusive


def walk_forward_splits(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    train_years: int = 3,
    test_years: int = 1,
    embargo_days: int = 30,
) -> list[Split]:
    """Expanding-window splits with an embargo gap.

    The embargo keeps event windows that straddle the boundary from
    leaking train information into test (an event on the last train day
    has a 20-trading-day return window reaching into test time).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    splits: list[Split] = []
    train_end = start + pd.DateOffset(years=train_years)
    while True:
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_start >= end:
            break
        splits.append(
            Split(start, train_end, test_start, min(test_end, end))
        )
        train_end = train_end + pd.DateOffset(years=test_years)
    return splits


# -- multiple testing ---------------------------------------------------
def benjamini_hochberg(pvals: pd.Series, q: float = 0.10) -> pd.DataFrame:
    """BH step-up procedure at FDR level q.

    Returns the input p-values with their rank, BH critical value, and
    a `reject` flag. Every spec in the grid gets a row — including the
    failures. The writeup reports all of them.
    """
    p = pvals.dropna().sort_values()
    m = len(p)
    ranks = np.arange(1, m + 1)
    crit = ranks / m * q
    below = p.values <= crit
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    out = pd.DataFrame(
        {"pval": p.values, "rank": ranks, "bh_crit": crit},
        index=p.index,
    )
    out["reject"] = out["rank"] <= k
    return out


# -- placebo ------------------------------------------------------------
def randomize_event_dates(
    events: pd.DataFrame,
    trading_days: pd.Series,
    rng: np.random.Generator | None = None,
    date_col: str = "event_date",
) -> pd.DataFrame:
    """Placebo: keep each event's ticker, replace its date with a
    uniformly random trading day. Rerunning the full evaluation on many
    placebo draws yields the null distribution your real result must
    beat. Any 'signal' that survives date randomization was never a
    signal — it was universe selection or methodology bias.
    """
    rng = rng or np.random.default_rng()
    out = events.copy()
    days = pd.to_datetime(trading_days).values
    out[date_col] = rng.choice(days, size=len(out))
    return out
