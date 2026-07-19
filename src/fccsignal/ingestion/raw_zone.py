"""Append-only raw zone.

Design goals, in order:

1. **Immutability is structural, not a convention.** Each ingestion
   batch is written to a brand-new file named by batch id. Nothing ever
   rewrites or deletes an existing file. "We never revise the raw zone"
   is then a property of the layout, not of programmer discipline.

2. **Idempotent pulls.** Every record gets a content hash over its
   business columns (ingestion_ts excluded). Re-pulling the same FCC
   grants tomorrow appends nothing. This makes schedulers safe to
   re-run and backfills safe to overlap.

3. **Auditability.** A manifest (one row per batch) records what was
   pulled, when, how many rows arrived and how many were new.

Storage format is pluggable ("parquet" for real use via pyarrow,
"csv" so the logic is testable in minimal environments).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fccsignal.contracts.pit import INGESTION_TS_COL

_HASH_COL = "_content_hash"
_BATCH_COL = "_batch_id"


def content_hash_frame(df: pd.DataFrame, business_cols: list[str]) -> pd.Series:
    """Deterministic per-row hash over the business columns only.

    Columns are sorted so hash stability doesn't depend on column order,
    and values are serialized via a canonical string form.
    """
    cols = sorted(business_cols)

    def _row_hash(row: pd.Series) -> str:
        payload = json.dumps(
            {c: (None if pd.isna(row[c]) else str(row[c])) for c in cols},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    return df.apply(_row_hash, axis=1)


class RawZone:
    """One raw-zone *table* (e.g. fcc_grants), stored as a directory of
    immutable batch files plus a manifest.
    """

    def __init__(self, root: Path | str, table: str, fmt: str = "parquet"):
        if fmt not in {"parquet", "csv"}:
            raise ValueError(f"unsupported fmt: {fmt}")
        self.fmt = fmt
        self.dir = Path(root) / table
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "_manifest.jsonl"

    # -- write ---------------------------------------------------------
    def append(self, df: pd.DataFrame, business_cols: list[str]) -> dict:
        """Append a pull. Returns the manifest entry for the batch.

        Rows whose content hash already exists in the table are dropped
        (idempotency). ingestion_ts is stamped here, once, in UTC —
        callers never supply it.
        """
        if df.empty:
            return self._log(batch_id=None, n_pulled=0, n_new=0)

        incoming = df.copy()
        incoming[_HASH_COL] = content_hash_frame(incoming, business_cols)

        seen = self._existing_hashes()
        fresh = incoming.loc[~incoming[_HASH_COL].isin(seen)].copy()

        batch_id = None
        if not fresh.empty:
            batch_id = uuid.uuid4().hex[:12]
            fresh[INGESTION_TS_COL] = pd.Timestamp(
                datetime.now(timezone.utc)
            ).tz_localize(None)
            fresh[_BATCH_COL] = batch_id
            path = self.dir / f"batch_{batch_id}.{self.fmt}"
            if path.exists():  # pragma: no cover - uuid collision guard
                raise FileExistsError(path)
            if self.fmt == "parquet":
                fresh.to_parquet(path, index=False)
            else:
                fresh.to_csv(path, index=False)

        return self._log(batch_id, n_pulled=len(incoming), n_new=len(fresh))

    # -- read ----------------------------------------------------------
    def read(self) -> pd.DataFrame:
        """Read the full table (all batches, concatenated).

        Downstream code must go through `pit_slice`/`AsOfView` before
        using this for any historical computation.
        """
        frames = []
        for path in sorted(self.dir.glob(f"batch_*.{self.fmt}")):
            if self.fmt == "parquet":
                frames.append(pd.read_parquet(path))
            else:
                frames.append(
                    pd.read_csv(path, parse_dates=[INGESTION_TS_COL])
                )
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # -- internals -----------------------------------------------------
    def _existing_hashes(self) -> set[str]:
        existing = self.read()
        if existing.empty or _HASH_COL not in existing.columns:
            return set()
        return set(existing[_HASH_COL])

    def _log(self, batch_id: str | None, n_pulled: int, n_new: int) -> dict:
        entry = {
            "batch_id": batch_id,
            "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_pulled": n_pulled,
            "n_new": n_new,
        }
        with self.manifest_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry
