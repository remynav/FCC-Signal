"""Calendar-time portfolio (Fama 1998 style).

Each trading day, hold (equal-weighted) every stock whose most recent
qualifying event occurred within the past `holding_days` trading days,
entering at t+`entry_lag`. Regress the portfolio's daily excess return
on the Fama-French factors; the intercept (alpha) is the signal's
risk-adjusted return, and — because overlapping events are absorbed
into a single portfolio time series — its t-stat is honest under the
event clustering that breaks the naive event-study t-stat.

Inference uses Newey-West (HAC) standard errors to tolerate residual
autocorrelation from the overlapping holding windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = ["mkt_rf", "smb", "hml"]


def build_portfolio_returns(
    events: pd.DataFrame,     # columns: ticker, event_date
    returns: pd.DataFrame,    # columns: date, ticker, ret
    factors: pd.DataFrame,    # provides the trading calendar + rf
    holding_days: int = 20,
    entry_lag: int = 2,
    min_names: int = 3,
) -> pd.DataFrame:
    """Daily equal-weighted portfolio excess returns.

    Days with fewer than `min_names` holdings are dropped rather than
    padded — thin-portfolio days are noise, and dropping them is the
    standard treatment (report the share dropped in the writeup).
    """
    fac = factors.sort_values("date").reset_index(drop=True)
    fac["date"] = pd.to_datetime(fac["date"])
    trading_days = fac["date"].reset_index(drop=True)

    rets = returns.copy()
    rets["date"] = pd.to_datetime(rets["date"])
    panel = rets.pivot(index="date", columns="ticker", values="ret")
    panel = panel.reindex(trading_days)

    # membership[t] = set of tickers held on trading day t
    in_window: dict[int, set] = {i: set() for i in range(len(trading_days))}
    for _, ev in events.iterrows():
        pos = int(
            np.searchsorted(
                trading_days.values, np.datetime64(pd.Timestamp(ev["event_date"]))
            )
        )
        start = pos + entry_lag
        for i in range(start, min(start + holding_days, len(trading_days))):
            in_window[i].add(ev["ticker"])

    rows = []
    rf = fac.set_index("date")["rf"]
    for i, day in enumerate(trading_days):
        names = [t for t in in_window[i] if t in panel.columns]
        if len(names) < min_names:
            continue
        r = panel.loc[day, names].dropna()
        if len(r) < min_names:
            continue
        rows.append(
            {"date": day, "port_exret": float(r.mean() - rf.loc[day]),
             "n_names": len(r)}
        )
    return pd.DataFrame(rows)


def newey_west_alpha(
    port: pd.DataFrame, factors: pd.DataFrame, lags: int = 5
) -> dict:
    """OLS of portfolio excess return on FF3 with HAC(lags) errors.

    Returns alpha (daily), its t-stat, and factor betas.
    """
    df = port.merge(factors, on="date", how="inner").dropna(
        subset=["port_exret"] + FACTORS
    )
    n = len(df)
    if n < 60:
        return {"alpha": np.nan, "t_alpha": np.nan, "n_days": n}

    y = df["port_exret"].values
    X = np.column_stack([np.ones(n)] + [df[f].values for f in FACTORS])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    # HAC covariance (Newey-West with Bartlett kernel)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))

    return {
        "alpha": float(beta[0]),
        "t_alpha": float(beta[0] / se[0]),
        "betas": {f: float(b) for f, b in zip(FACTORS, beta[1:])},
        "n_days": n,
    }
