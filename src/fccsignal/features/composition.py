"""Composition features: first-in-class entry, class-mix entropy,
confidentiality intensity.

All three are computed per event or per (ticker, date) using only
filings dated strictly before the observation — same PIT discipline as
velocity.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def first_in_class(
    events: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "grant_date",
    class_col: str = "equipment_class",
) -> pd.DataFrame:
    """Flag filings in an equipment class the company has never filed
    in before (a product-line-expansion signal, not more-of-the-same).

    Ties on the same day: a company's several same-day filings in a
    brand-new class are all flagged (they jointly constitute the entry).
    """
    ev = events.copy()
    ev[date_col] = pd.to_datetime(ev[date_col])
    ev = ev.sort_values([ticker_col, date_col], kind="stable")

    first_seen = (
        ev.groupby([ticker_col, class_col])[date_col].transform("min")
    )
    ev["first_in_class"] = ev[date_col] == first_seen
    # A company's very first-ever filing day flags everything trivially;
    # keep the flag but expose tenure so specs can require history.
    tenure = ev.groupby(ticker_col)[date_col].transform("min")
    ev["days_since_first_filing"] = (ev[date_col] - tenure).dt.days
    return ev


def class_mix_entropy(
    events: pd.DataFrame,
    as_of: pd.Timestamp | str,
    lookback_days: int = 365,
    ticker_col: str = "ticker",
    date_col: str = "grant_date",
    class_col: str = "equipment_class",
) -> pd.DataFrame:
    """Shannon entropy of a company's equipment-class mix over the
    trailing window ending at (and excluding) `as_of`.

    High entropy = diversified product activity; a jump in entropy is a
    diversification event.
    """
    as_of = pd.Timestamp(as_of)
    ev = events.copy()
    ev[date_col] = pd.to_datetime(ev[date_col])
    window = ev.loc[
        (ev[date_col] < as_of)
        & (ev[date_col] >= as_of - pd.Timedelta(days=lookback_days))
    ]
    rows = []
    for tkr, grp in window.groupby(ticker_col):
        p = grp[class_col].value_counts(normalize=True).values
        entropy = float(-(p * np.log(p)).sum()) if len(p) else np.nan
        rows.append({"ticker": tkr, "as_of": as_of, "class_entropy": entropy})
    return pd.DataFrame(rows)


def confidentiality_intensity(
    events: pd.DataFrame,
    as_of: pd.Timestamp | str,
    lookback_days: int = 180,
    ticker_col: str = "ticker",
    date_col: str = "grant_date",
    conf_col: str = "has_confidentiality",
) -> pd.DataFrame:
    """Share of a company's trailing filings that carried a short-term
    confidentiality request (photos/manuals withheld).

    Interpretation: the company is guarding an unannounced product. The
    request itself is public metadata even when the exhibits aren't.
    """
    as_of = pd.Timestamp(as_of)
    ev = events.copy()
    ev[date_col] = pd.to_datetime(ev[date_col])
    window = ev.loc[
        (ev[date_col] < as_of)
        & (ev[date_col] >= as_of - pd.Timedelta(days=lookback_days))
    ]
    grp = window.groupby(ticker_col)[conf_col]
    out = grp.mean().rename("conf_intensity").reset_index()
    out["n_filings_window"] = grp.size().values
    out["as_of"] = as_of
    return out
