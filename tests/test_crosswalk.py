import pandas as pd
import pytest

from fccsignal.entity.crosswalk import Crosswalk, CrosswalkError


def _acquisition_xwalk():
    # Private co "2ABCD" acquired by PUBL effective 2022-03-01
    return Crosswalk(pd.DataFrame({
        "grantee_code": ["2ABCD", "2ABCD", "XYZ"],
        "ticker":       ["",      "PUBL",  "XYZC"],
        "valid_from":   ["2015-01-01", "2022-03-01", "2010-01-01"],
        "valid_to":     ["2022-03-01", None,          None],
        "source":       ["manual", "manual", "edgar_fuzzy"],
        "confidence":   [1.0, 1.0, 0.9],
    }))


def test_pre_acquisition_filings_do_not_map_to_future_parent():
    xw = _acquisition_xwalk()
    assert xw.resolve("2ABCD", "2019-06-15") == ""      # known unmapped
    assert xw.resolve("2ABCD", "2023-01-10") == "PUBL"  # post-acquisition


def test_boundary_semantics_from_inclusive_to_exclusive():
    xw = _acquisition_xwalk()
    assert xw.resolve("2ABCD", "2022-02-28") == ""
    assert xw.resolve("2ABCD", "2022-03-01") == "PUBL"


def test_unknown_code_returns_none():
    assert _acquisition_xwalk().resolve("NOPE", "2020-01-01") is None


def test_overlapping_intervals_rejected():
    with pytest.raises(CrosswalkError):
        Crosswalk(pd.DataFrame({
            "grantee_code": ["AAA", "AAA"],
            "ticker": ["X", "Y"],
            "valid_from": ["2020-01-01", "2021-06-01"],
            "valid_to":   ["2022-01-01", None],
            "source": ["manual", "manual"],
            "confidence": [1.0, 1.0],
        }))


def test_inverted_interval_rejected():
    with pytest.raises(CrosswalkError):
        Crosswalk(pd.DataFrame({
            "grantee_code": ["AAA"], "ticker": ["X"],
            "valid_from": ["2022-01-01"], "valid_to": ["2020-01-01"],
            "source": ["manual"], "confidence": [1.0],
        }))


def test_coverage_report_counts():
    xw = _acquisition_xwalk()
    events = pd.DataFrame({
        "grantee_code": ["2ABCD", "2ABCD", "XYZ", "NOPE"],
        "grant_date": ["2019-01-01", "2023-01-01", "2020-01-01", "2020-01-01"],
    })
    rep = xw.coverage_report(events)
    assert rep == {
        "n_events": 4, "mapped": 2, "known_unmapped": 1,
        "unknown": 1, "mapped_share": 0.5,
    }
