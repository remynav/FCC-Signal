import numpy as np
import pandas as pd

from fccsignal.evaluation.calendar_time import (
    build_portfolio_returns, newey_west_alpha,
)
from fccsignal.evaluation.event_study import EventStudyConfig, run_event_study
from fccsignal.evaluation.protocol import (
    benjamini_hochberg, randomize_event_dates, walk_forward_splits,
)


def _market(n_days=600, seed=3, n_tickers=6, inject=None):
    """Synthetic factor + return panel; `inject` = dict(ticker->(pos, bump))
    adds abnormal return at trading-day positions."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    fac = pd.DataFrame({
        "date": dates,
        "mkt_rf": rng.normal(0.0003, 0.01, n_days),
        "smb": rng.normal(0, 0.005, n_days),
        "hml": rng.normal(0, 0.005, n_days),
        "rf": 0.0001,
    })
    rows = []
    for i in range(n_tickers):
        tkr = f"T{i}"
        beta = 1.0 + 0.1 * i
        eps = rng.normal(0, 0.008, n_days)
        ret = fac["rf"].values + beta * fac["mkt_rf"].values + eps
        if inject and tkr in inject:
            pos, bump = inject[tkr]
            ret[pos: pos + 5] += bump
        rows.append(pd.DataFrame({"date": dates, "ticker": tkr, "ret": ret}))
    return fac, pd.concat(rows, ignore_index=True)


def test_event_study_recovers_injected_car():
    inject = {"T0": (302, 0.02)}  # +2% for 5 days = ~10% injected CAR
    fac, rets = _market(inject=inject)
    events = pd.DataFrame({
        "ticker": ["T0"], "event_date": [fac["date"].iloc[300]],
    })
    cfg = EventStudyConfig(entry_lag=2)
    study = run_event_study(events, rets, fac, cfg)
    # ~10% injected; event-window noise std ~ 0.8%*sqrt(21) ~ 3.7%,
    # and the seed is fixed, so > 0.05 is a stable, meaningful bound.
    assert study["car"].iloc[0] > 0.05


def test_event_study_insufficient_history_yields_nan():
    fac, rets = _market()
    events = pd.DataFrame({
        "ticker": ["T0"], "event_date": [fac["date"].iloc[50]],
    })
    study = run_event_study(events, rets, fac)
    assert np.isnan(study["car"].iloc[0])


def test_calendar_time_alpha_positive_when_events_precede_gains():
    rng = np.random.default_rng(11)
    inject = {f"T{i}": (250 + 40 * i, 0.008) for i in range(4)}
    fac, rets = _market(inject=inject, seed=5)
    events = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(4)],
        "event_date": [fac["date"].iloc[248 + 40 * i] for i in range(4)],
    })
    port = build_portfolio_returns(events, rets, fac, holding_days=20,
                                   entry_lag=2, min_names=1)
    res = newey_west_alpha(port, fac)
    assert res["n_days"] >= 60
    assert res["alpha"] > 0


def test_walk_forward_has_embargo_and_no_overlap():
    splits = walk_forward_splits("2015-01-01", "2024-01-01",
                                 train_years=3, test_years=1,
                                 embargo_days=30)
    assert len(splits) >= 4
    for s in splits:
        assert (s.test_start - s.train_end).days >= 30
        assert s.train_end <= s.test_start < s.test_end


def test_benjamini_hochberg_known_example():
    p = pd.Series(
        [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216],
        index=[f"spec{i}" for i in range(10)],
    )
    out = benjamini_hochberg(p, q=0.10)
    # Hand-computed: criticals are k/10*0.10 = .01,...,.10; the largest
    # k with p_(k) <= crit_(k) is k=6 (0.060 <= 0.060), so BH rejects
    # the SIX smallest — including .039 and .041, which individually
    # exceed their criticals. That non-monotone pattern is the step-up
    # property this test pins down.
    assert out["reject"].sum() == 6
    assert bool(out.loc["spec5", "reject"]) is True
    assert bool(out.loc["spec6", "reject"]) is False


def test_placebo_preserves_tickers_and_uses_trading_days():
    fac, _ = _market()
    events = pd.DataFrame({
        "ticker": ["T0", "T1"],
        "event_date": [fac["date"].iloc[100], fac["date"].iloc[200]],
    })
    rng = np.random.default_rng(0)
    plc = randomize_event_dates(events, fac["date"], rng=rng)
    assert list(plc["ticker"]) == ["T0", "T1"]
    assert plc["event_date"].isin(fac["date"].values).all()
