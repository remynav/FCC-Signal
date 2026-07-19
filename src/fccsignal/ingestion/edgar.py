"""SEC EDGAR puller (skeleton) — fuel for entity resolution.

Pulls: company tickers/CIK master file, and Exhibit-21 subsidiary
lists from 10-K filings (subsidiary legal names are how grantee-code
applicant names get attached to public parents).

Note EDGAR's fair-access policy: identify with a User-Agent contact
string and stay under the published request-rate limit.
"""

from __future__ import annotations

import pandas as pd

BUSINESS_COLS_COMPANIES = ["cik", "ticker", "company_name"]
BUSINESS_COLS_SUBSIDIARIES = ["cik", "filing_date", "subsidiary_name"]


def pull_company_master() -> pd.DataFrame:  # pragma: no cover - network
    raise NotImplementedError


def pull_ex21_subsidiaries(cik: str) -> pd.DataFrame:  # pragma: no cover
    raise NotImplementedError
