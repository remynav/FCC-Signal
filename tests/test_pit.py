import pandas as pd
import pytest

from fccsignal.contracts.pit import (
    AsOfView, PITViolationError, assert_pit_clean, pit_slice,
)


def _df():
    return pd.DataFrame({
        "x": [1, 2, 3],
        "ingestion_ts": pd.to_datetime(
            ["2024-01-01", "2024-06-01", "2025-01-01"]
        ),
    })


def test_pit_slice_excludes_future_rows():
    out = pit_slice(_df(), "2024-06-30")
    assert list(out["x"]) == [1, 2]


def test_pit_slice_boundary_is_inclusive():
    out = pit_slice(_df(), "2024-06-01")
    assert list(out["x"]) == [1, 2]


def test_missing_ts_col_refuses_to_serve():
    with pytest.raises(PITViolationError):
        pit_slice(pd.DataFrame({"x": [1]}), "2024-01-01")


def test_assert_pit_clean_trips_on_leak():
    with pytest.raises(PITViolationError):
        assert_pit_clean(_df(), "2024-06-30")
    assert_pit_clean(_df(), "2025-01-01")  # no raise


def test_asof_view_prefilters():
    v = AsOfView(_df(), pd.Timestamp("2024-06-30"))
    assert len(v.read()) == 2
