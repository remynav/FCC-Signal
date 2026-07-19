"""FCC EAS puller (skeleton).

Sources, in preference order:
1. Bulk files / open data (opendata.fcc.gov EAS datasets) for backfill.
2. OET EAS web services (getFCCIDList et al., see FCC KDB 953436) for
   incremental daily pulls.
3. fccid.io as a cross-check mirror only — never the primary source.

Contract with the raw zone: pullers return *raw, unmodified* frames
plus the list of business columns for hashing. No parsing, no joins,
no normalization — that belongs downstream, so raw always preserves
what the source actually said.

Business columns to capture per grant (minimum viable set):
    fcc_id, grantee_code, product_code, applicant_name, grant_date,
    equipment_class, rule_parts, has_confidentiality (from the
    confidentiality request metadata), application_purpose
"""

from __future__ import annotations

import pandas as pd

BUSINESS_COLS = [
    "fcc_id", "grantee_code", "product_code", "applicant_name",
    "grant_date", "equipment_class", "rule_parts",
    "has_confidentiality", "application_purpose",
]


def pull_daily(date: str) -> pd.DataFrame:  # pragma: no cover - network
    """Fetch grants issued on `date`. TODO: implement against the OET
    EAS API; respect rate limits; return raw frame with BUSINESS_COLS."""
    raise NotImplementedError


def pull_backfill(start: str, end: str) -> pd.DataFrame:  # pragma: no cover
    """Backfill from bulk files. IMPORTANT: rows ingested via backfill
    share one ingestion_ts (the pull time), which is honest — we truly
    learned them all at once. The evaluation layer's entry-lag
    sensitivity exists precisely because of this."""
    raise NotImplementedError
