"""Transaction costs and capacity (skeleton).

Cost model: half-spread + square-root impact.
    cost_bps = half_spread_bps + k * sqrt(trade_size / ADV) * sigma
with k calibrated conservatively (literature values ~0.5-1.0 for the
square-root law; document the choice and show results at 2x costs).

Capacity: strategy AUM at which expected alpha net of impact hits
zero, given max participation (e.g. 5% of ADV) per name. Small/mid-cap
hardware is illiquid; report capacity honestly — a true but small edge
is a fine research result and a better interview story than an
implausibly large one.
"""

from __future__ import annotations


def sqrt_impact_bps(trade_dollars: float, adv_dollars: float,
                    daily_vol: float, k: float = 0.75) -> float:
    if adv_dollars <= 0:
        return float("inf")
    return 1e4 * k * daily_vol * (trade_dollars / adv_dollars) ** 0.5
