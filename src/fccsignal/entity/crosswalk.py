"""Effective-dated entity crosswalk: grantee_code -> ticker, *as of a date*.

Why effective-dated
-------------------
Entities move. A private company files under its own grantee code in
2019 and is acquired by a public parent in 2022. Its 2019 filings must
NOT map to that parent — in 2019 the information was not attributable
to any tradeable ticker. A static mapping silently launders future
knowledge (the acquisition) into the past. Every mapping row therefore
carries [valid_from, valid_to) and resolution takes a date.

Schema (one row per mapping interval)
-------------------------------------
    grantee_code   str    FCC grantee code (3 or 5 chars)
    ticker         str    parent's ticker, or the sentinel "" meaning
                          "known unmapped" (private co, contract mfr)
    valid_from     date   inclusive
    valid_to       date   exclusive; NaT = open-ended / current
    source         str    "edgar_fuzzy" | "ex21" | "manual"
    confidence     float  [0, 1]; manual overrides carry 1.0
    note           str    free text, required for manual rows

Invariants (validated on load)
------------------------------
* Intervals for the same grantee_code never overlap.
* valid_from < valid_to whenever valid_to is present.
* Manual override rows win over automated rows on conflict — enforced
  upstream by the builder, and re-checked here as non-overlap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLS = [
    "grantee_code",
    "ticker",
    "valid_from",
    "valid_to",
    "source",
    "confidence",
]


class CrosswalkError(ValueError):
    pass


class Crosswalk:
    def __init__(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise CrosswalkError(f"crosswalk missing columns: {missing}")
        df = df.copy()
        df["valid_from"] = pd.to_datetime(df["valid_from"])
        df["valid_to"] = pd.to_datetime(df["valid_to"])  # NaT allowed
        self._validate(df)
        self.df = df

    # -- validation ----------------------------------------------------
    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        bad = df.dropna(subset=["valid_to"])
        bad = bad.loc[bad["valid_from"] >= bad["valid_to"]]
        if not bad.empty:
            raise CrosswalkError(
                f"{len(bad)} row(s) with valid_from >= valid_to, e.g. "
                f"grantee_code={bad.iloc[0]['grantee_code']!r}"
            )

        for code, grp in df.groupby("grantee_code"):
            grp = grp.sort_values("valid_from")
            prev_end = None
            for _, row in grp.iterrows():
                if prev_end is not None and (
                    pd.isna(prev_end) or row["valid_from"] < prev_end
                ):
                    raise CrosswalkError(
                        f"overlapping intervals for grantee_code={code!r}"
                    )
                prev_end = row["valid_to"]

    # -- resolution ----------------------------------------------------
    def resolve(self, grantee_code: str, on_date: pd.Timestamp | str) -> str | None:
        """Ticker for `grantee_code` as of `on_date`.

        Returns:
            ticker string  -> mapped to a public parent on that date
            ""             -> known and deliberately unmapped (private,
                              contract manufacturer, foreign-only, ...)
            None           -> unknown grantee code / uncovered date
        """
        on_date = pd.Timestamp(on_date)
        rows = self.df.loc[self.df["grantee_code"] == grantee_code]
        for _, row in rows.iterrows():
            starts_ok = row["valid_from"] <= on_date
            ends_ok = pd.isna(row["valid_to"]) or on_date < row["valid_to"]
            if starts_ok and ends_ok:
                return row["ticker"]
        return None

    def resolve_frame(
        self,
        events: pd.DataFrame,
        code_col: str = "grantee_code",
        date_col: str = "grant_date",
    ) -> pd.DataFrame:
        """Vector-ish resolution for an events frame; adds a `ticker`
        column resolved as of each event's own date."""
        out = events.copy()
        out["ticker"] = [
            self.resolve(c, d) for c, d in zip(out[code_col], out[date_col])
        ]
        return out

    # -- io --------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: Path | str) -> "Crosswalk":
        return cls(pd.read_csv(path))

    def coverage_report(self, events: pd.DataFrame) -> dict:
        """Share of events resolved / known-unmapped / unknown — a
        first-class quality metric for the writeup."""
        resolved = self.resolve_frame(events)
        n = len(resolved)
        mapped = int((resolved["ticker"].notna() & (resolved["ticker"] != "")).sum())
        known_unmapped = int((resolved["ticker"] == "").sum())
        return {
            "n_events": n,
            "mapped": mapped,
            "known_unmapped": known_unmapped,
            "unknown": n - mapped - known_unmapped,
            "mapped_share": mapped / n if n else float("nan"),
        }
