"""Factor-adjusted event study.

Methodology
-----------
For each event (ticker, event_date):

1. Estimation window: trading days [-141, -21] relative to the event
   (~120 obs), deliberately ending 20 trading days BEFORE the event so
   pre-announcement drift cannot contaminate the "normal" return model.
2. Fit r_i - rf = a + b*MKT + s*SMB + h*HML + e by OLS on that window.
3. Event window: abnormal return AR_t = (r_i - rf) - (a + b*MKT + s*SMB
   + h*HML); CAR = sum of AR over [t_start, t_end].
4. Execution-lag honesty: `entry_lag` shifts the event anchor to the
   t+`entry_lag` trading day; run the whole study at lags 1..5 and
   report the curve. A signal alive only at t+0 was never tradeable.

Statistics caveat (by design): the naive cross-sectional t-stat over
CARs assumes independent events. FCC filings cluster in time, so that
assumption fails; treat the t-stat here as descriptive and rely on
`calendar_time.py` for inference that survives cross-sectional
correlation. This split of responsibilities is intentional and should
be stated in the writeup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FACTORS = ["mkt_rf", "smb", "hml"]


@dataclass
class EventStudyConfig:
    est_start: int = -141   # trading-day offsets relative to event
    est_end: int = -21      # exclusive of event; 20-day gap
    evt_start: int = 0
    evt_end: int = 20
    entry_lag: int = 2      # trade at t+2 by default
    min_est_obs: int = 90


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def run_event_study(
    events: pd.DataFrame,
    returns: pd.DataFrame,   # columns: date, ticker, ret
    factors: pd.DataFrame,   # columns: date, mkt_rf, smb, hml, rf
    cfg: EventStudyConfig | None = None,
) -> pd.DataFrame:
    """One row per event with its CAR; NaN CAR when data insufficient.

    All alignment is on TRADING days (the factor calendar), not
    calendar days — weekends/holidays never enter a window.
    """
    cfg = cfg or EventStudyConfig()
    fac = factors.sort_values("date").reset_index(drop=True)
    fac["date"] = pd.to_datetime(fac["date"])
    trading_days = fac["date"]
    day_index = {d: i for i, d in enumerate(trading_days)}

    rets = returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    panel = rets.pivot(index="date", columns="ticker", values="ret")
    panel = panel.reindex(trading_days)

    rows = []
    for _, ev in events.iterrows():
        tkr = ev["ticker"]
        d0 = pd.Timestamp(ev["event_date"])
        # anchor = first trading day >= event date, then apply entry lag
        pos = int(np.searchsorted(trading_days.values, np.datetime64(d0)))
        anchor = pos + cfg.entry_lag
        est_lo, est_hi = anchor + cfg.est_start, anchor + cfg.est_end
        evt_lo, evt_hi = anchor + cfg.evt_start, anchor + cfg.evt_end
        if est_lo < 0 or evt_hi >= len(trading_days) or tkr not in panel:
            rows.append({**ev.to_dict(), "car": np.nan, "n_est": 0})
            continue

        r = panel[tkr]
        est_dates = trading_days.iloc[est_lo:est_hi]
        est_ret = r.loc[est_dates]
        est_fac = fac.iloc[est_lo:est_hi]
        ok = est_ret.notna()
        if int(ok.sum()) < cfg.min_est_obs:
            rows.append({**ev.to_dict(), "car": np.nan, "n_est": int(ok.sum())})
            continue

        y = (est_ret[ok].values - est_fac.loc[ok.values, "rf"].values)
        X = np.column_stack(
            [np.ones(int(ok.sum()))] + [est_fac.loc[ok.values, f].values for f in FACTORS]
        )
        beta = _ols(y, X)

        evt_dates = trading_days.iloc[evt_lo : evt_hi + 1]
        evt_ret = r.loc[evt_dates]
        evt_fac = fac.iloc[evt_lo : evt_hi + 1]
        ok_e = evt_ret.notna()
        Xe = np.column_stack(
            [np.ones(int(ok_e.sum()))]
            + [evt_fac.loc[ok_e.values, f].values for f in FACTORS]
        )
        expected = Xe @ beta
        ar = (evt_ret[ok_e].values - evt_fac.loc[ok_e.values, "rf"].values) - expected
        rows.append(
            {**ev.to_dict(), "car": float(np.nansum(ar)), "n_est": int(ok.sum())}
        )
    return pd.DataFrame(rows)


def naive_car_tstat(study: pd.DataFrame) -> dict:
    """Descriptive only — see module docstring for why this t-stat is
    optimistic under event clustering."""
    cars = study["car"].dropna()
    n = len(cars)
    if n < 2:
        return {"mean_car": np.nan, "t": np.nan, "n": n}
    t = cars.mean() / (cars.std(ddof=1) / np.sqrt(n))
    return {"mean_car": float(cars.mean()), "t": float(t), "n": n}
