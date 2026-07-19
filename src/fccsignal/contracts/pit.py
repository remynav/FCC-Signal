"""Point-in-time (PIT) contract.

The single rule this repo is built around:

    Any feature computed "as of" date T may use only rows whose
    ingestion_ts is <= T.

This module makes that rule *structural* rather than aspirational.
Downstream code should never filter on ingestion_ts by hand; it should
go through `pit_slice` / `AsOfView` so violations are impossible to
write silently and easy to test for.

Why ingestion_ts and not the record's own event date?
------------------------------------------------------
A record's business date (e.g. an FCC grant date) says when something
*happened*. ingestion_ts says when *we could have known about it*. Only
the latter is a defensible basis for a backtest. For backfilled history
the two are conflated (we ingested years of grants in one pull), which
is exactly why the evaluation layer trades at t+2 relative to the
business date and runs a lag-sensitivity analysis: see
`fccsignal.evaluation.event_study`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

INGESTION_TS_COL = "ingestion_ts"


class PITViolationError(RuntimeError):
    """Raised when code attempts to read data not yet visible as-of T."""


def _require_ts_col(df: pd.DataFrame, ts_col: str) -> None:
    if ts_col not in df.columns:
        raise PITViolationError(
            f"DataFrame has no '{ts_col}' column; refusing to serve it "
            "point-in-time. Every raw-zone table must carry an ingestion "
            "timestamp."
        )


def pit_slice(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    ts_col: str = INGESTION_TS_COL,
) -> pd.DataFrame:
    """Return only the rows visible on or before `as_of`.

    This is the only sanctioned way to read raw/feature data for a
    historical computation.
    """
    _require_ts_col(df, ts_col)
    as_of = pd.Timestamp(as_of)
    ts = pd.to_datetime(df[ts_col])
    return df.loc[ts <= as_of].copy()


def assert_pit_clean(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    ts_col: str = INGESTION_TS_COL,
) -> None:
    """Assert that `df` contains nothing invisible at `as_of`.

    Use in tests and as a tripwire at layer boundaries: features hand
    the evaluation layer a frame, evaluation asserts it's clean.
    """
    _require_ts_col(df, ts_col)
    as_of = pd.Timestamp(as_of)
    ts = pd.to_datetime(df[ts_col])
    n_bad = int((ts > as_of).sum())
    if n_bad:
        raise PITViolationError(
            f"{n_bad} row(s) have {ts_col} after as_of={as_of.date()}; "
            "a future leak reached this layer."
        )


@dataclass(frozen=True)
class AsOfView:
    """A read-only, as-of-dated view over a table.

    Feature code receives an AsOfView, not a raw DataFrame, so the
    as_of date travels with the data and every read is pre-filtered.
    """

    df: pd.DataFrame
    as_of: pd.Timestamp
    ts_col: str = INGESTION_TS_COL

    def __post_init__(self) -> None:  # validate eagerly, fail loudly
        _require_ts_col(self.df, self.ts_col)

    def read(self) -> pd.DataFrame:
        return pit_slice(self.df, self.as_of, self.ts_col)
