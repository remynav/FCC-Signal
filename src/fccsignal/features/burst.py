"""Filing-burst detection via Poisson surprise.

Instead of an arbitrary z-score threshold, treat each company's filing
arrivals as a Poisson process and ask: given this company's OWN
trailing baseline rate, how surprising is the count we just observed?

    surprise(T) = -log10 P( X >= x_T )   with  X ~ Poisson(lambda_T)

where lambda_T is the expected count over the observation window,
estimated from a strictly-prior baseline window. surprise >= 3 means
"under this company's own history, a burst this large happens with
p <= 1e-3." The event definition is therefore company-relative and
statistically principled — the defensible answer to "how did you
define an event?"

Implementation note: counts are small integers, so the Poisson upper
tail is computed by direct summation of the pmf — no scipy needed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam).

    Numerical note: computing 1 - CDF loses all precision once the tail
    drops below float64 epsilon (~1e-16), which matters because the
    surprise statistic is -log10 of exactly this tail. So the tail is
    summed DIRECTLY from k upward — the terms decay geometrically for
    k > lam, so a few hundred terms are always enough.
    """
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    log_pmf_k = -lam + k * math.log(lam) - math.lgamma(k + 1)
    term = math.exp(log_pmf_k)  # underflows to 0.0 only when tail ~ 0
    total = 0.0
    i = k
    while term > total * 1e-17 + 1e-320 and i < k + 10_000:
        total += term
        i += 1
        term *= lam / i
    return min(total, 1.0)


def burst_surprise(
    counts: pd.DataFrame,
    obs_window: int = 30,
    baseline_window: int = 365,
    min_history: int = 180,
) -> pd.DataFrame:
    """Per (ticker, date): -log10 tail probability of the trailing
    `obs_window`-day count under the company's strictly-prior baseline
    rate, scaled to the observation window.

    Requires `counts` from features.velocity.daily_counts.
    """
    frames = []
    for tkr, grp in counts.groupby("ticker"):
        g = grp.sort_values("date").set_index("date")
        x = g["n_filings"].rolling(obs_window, min_periods=1).sum()
        prior_daily = g["n_filings"].shift(1)  # strictly prior
        base_rate = prior_daily.rolling(
            baseline_window, min_periods=min_history
        ).mean()
        lam = base_rate * obs_window  # expected count in obs window
        surprise = [
            (
                -math.log10(max(poisson_sf(int(k), l), 1e-300))
                if not (np.isnan(l) or np.isnan(k))
                else np.nan
            )
            for k, l in zip(x.values, lam.values)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "ticker": tkr,
                    "date": g.index,
                    "obs_count": x.values,
                    "expected_count": lam.values,
                    "burst_surprise": surprise,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
