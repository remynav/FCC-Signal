import pandas as pd

from fccsignal.ingestion.raw_zone import RawZone


def _grants():
    return pd.DataFrame({
        "fcc_id": ["2ABCD-W1", "2ABCD-W2"],
        "grant_date": ["2024-05-01", "2024-05-02"],
    })


def test_append_then_reread(tmp_path):
    rz = RawZone(tmp_path, "fcc_grants", fmt="csv")
    entry = rz.append(_grants(), business_cols=["fcc_id", "grant_date"])
    assert entry["n_new"] == 2
    df = rz.read()
    assert len(df) == 2
    assert "ingestion_ts" in df.columns


def test_repull_is_idempotent(tmp_path):
    rz = RawZone(tmp_path, "fcc_grants", fmt="csv")
    rz.append(_grants(), business_cols=["fcc_id", "grant_date"])
    entry = rz.append(_grants(), business_cols=["fcc_id", "grant_date"])
    assert entry["n_new"] == 0
    assert len(rz.read()) == 2


def test_new_rows_append_without_touching_old_files(tmp_path):
    rz = RawZone(tmp_path, "fcc_grants", fmt="csv")
    rz.append(_grants(), business_cols=["fcc_id", "grant_date"])
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "fcc_grants").glob("batch_*.csv")}

    more = pd.DataFrame({"fcc_id": ["2ABCD-W3"], "grant_date": ["2024-06-01"]})
    rz.append(more, business_cols=["fcc_id", "grant_date"])

    after = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "fcc_grants").glob("batch_*.csv")}
    for name, mtime in before.items():
        assert after[name] == mtime  # old batch files untouched
    assert len(after) == len(before) + 1
    assert len(rz.read()) == 3


def test_manifest_logs_every_pull(tmp_path):
    rz = RawZone(tmp_path, "fcc_grants", fmt="csv")
    rz.append(_grants(), business_cols=["fcc_id", "grant_date"])
    rz.append(_grants(), business_cols=["fcc_id", "grant_date"])
    lines = (tmp_path / "fcc_grants" / "_manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
