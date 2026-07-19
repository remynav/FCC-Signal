import numpy as np
import pandas as pd

from fccsignal.features.burst import burst_surprise, poisson_sf
from fccsignal.features.composition import first_in_class
from fccsignal.features.velocity import daily_counts, velocity_features


def test_poisson_sf_sane():
    assert poisson_sf(0, 5.0) == 1.0
    assert abs(poisson_sf(1, 1.0) - (1 - np.exp(-1))) < 1e-12
    assert poisson_sf(30, 1.0) < 1e-20


def test_burst_detector_flags_injected_burst():
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    rng = np.random.default_rng(7)
    n = rng.poisson(0.05, size=800)          # quiet baseline
    n[700:715] += 3                           # injected burst
    events = pd.DataFrame({
        "ticker": "TEST",
        "grant_date": np.repeat(dates, n),
    })
    counts = daily_counts(events)
    bs = burst_surprise(counts)
    in_burst = bs.loc[bs["date"].between(dates[705], dates[720]), "burst_surprise"]
    pre_burst = bs.loc[bs["date"].between(dates[400], dates[690]), "burst_surprise"]
    assert in_burst.max() > 3.0
    assert pre_burst.max() < 3.0


def test_velocity_z_uses_strictly_prior_baseline():
    # Constant filing rate then a jump: z must be driven by the jump
    # relative to PRIOR history, and must be NaN before min_history.
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    events = pd.DataFrame({
        "ticker": "TEST",
        "grant_date": list(dates[::10]) + list(dates[390:400]),
    })
    counts = daily_counts(events)
    feats = velocity_features(counts, short_window=30, long_window=180,
                              min_history=90)
    early = feats.loc[feats["date"] < dates[90], "accel_z"]
    assert early.isna().all()
    assert feats["accel_z"].iloc[-1] > 1.0


def test_first_in_class_flags_new_class_only():
    events = pd.DataFrame({
        "ticker": ["A", "A", "A", "A"],
        "grant_date": ["2020-01-01", "2021-01-01", "2021-01-01", "2022-01-01"],
        "equipment_class": ["DTS", "DTS", "PCE", "PCE"],
    })
    out = first_in_class(events)
    flags = out.sort_values(["grant_date", "equipment_class"])["first_in_class"]
    assert list(flags) == [True, False, True, False]
