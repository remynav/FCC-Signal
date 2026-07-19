# fcc-signal

Research pipeline testing whether FCC equipment-authorization filings act as
a leading indicator of hardware product cycles, and whether that indicator
predicts forward abnormal returns in small/mid-cap hardware stocks.

Every RF-emitting device sold in the US must receive an FCC grant of
certification before it can be marketed or imported. Grants are public,
timestamped, and typically precede retail launch. For a small or mid-cap
hardware company, a single product cycle moves the P&L, so filing dynamics
plausibly carry information the market has not fully priced. This repo is
built to test that claim honestly, including the strong possibility that the
answer is "not tradeably."

## The point-in-time contract

The rule the whole repo is built around: any feature computed as of date T
may use only rows whose `ingestion_ts` is on or before T.

The rule is structural, not aspirational:

* `contracts/pit.py` is the only sanctioned read path for historical
  computation (`pit_slice`, `AsOfView`, `assert_pit_clean` as a tripwire at
  layer boundaries).
* `ingestion/raw_zone.py` is append-only by construction. Each pull writes a
  new immutable batch file; content hashing makes re-pulls idempotent; a
  manifest logs every pull. Raw is never edited, so any feature store state
  is rebuildable as of any historical date.
* `entity/crosswalk.py` is effective-dated. Every grantee-code-to-ticker
  mapping carries `valid_from`/`valid_to`, so an entity acquired in 2022
  does not map to its future parent in 2019. Overlap and inverted intervals
  are rejected at load.

One honest limitation is stated up front: for backfilled history, the grant
date is known but the moment the record became publicly visible is not
provable. The evaluation layer therefore trades at t+2 by default and runs
the whole study at entry lags 1 through 5. A signal alive only at t+0 was
never tradeable, and the writeup will say so if that is what we find.

## Layout

    configs/
      spec_grid.yaml        pre-registered specification grid (commit = registration)
      pipeline.yaml         schedules, storage roots
    src/fccsignal/
      contracts/pit.py      PIT enforcement
      ingestion/            raw zone + FCC / EDGAR / market-data pullers
      entity/               effective-dated crosswalk + manual overrides
      features/             velocity, burst (Poisson surprise), first-in-class,
                            class-mix entropy, confidentiality intensity
      universe/             small/mid-cap hardware membership, survivorship-free
      evaluation/           event study, calendar-time portfolio, walk-forward,
                            Benjamini-Hochberg, placebo, cost/capacity model
    tests/                  the discipline, enforced

## Methodology in one paragraph

Events are defined per the pre-registered grid (burst surprise, acceleration
z, first-in-class entry, confidentiality jumps). Abnormal returns come from a
Fama-French three-factor event study whose estimation window ends 20 trading
days before the event so pre-announcement drift cannot contaminate the normal
return model. Because filings cluster in time, the naive cross-sectional
t-stat is treated as descriptive only; inference rests on a calendar-time
portfolio with Newey-West errors, which absorbs overlapping events by
construction. Validation is walk-forward with an embargo gap. All p-values
across the grid pass through Benjamini-Hochberg at q = 0.10, and the real
result must beat a null distribution built from randomized-event-date placebo
runs. Costs use half-spread plus square-root impact, with capacity reported.

## Running

    make install     # pip install -e ".[dev]"
    make test        # pytest
    make pull        # scheduled ingestion (Prefect entry point, TODO)
    make features    # rebuild PIT feature store from raw (TODO)
    make evaluate    # run the pre-registered grid (TODO)

Tests run today; the TODO entry points are the build roadmap below.

## Roadmap

1. Implement `ingestion/fcc_eas.py` backfill from FCC bulk data, then the
   daily incremental pull. Wire into Prefect.
2. Implement EDGAR pulls and the crosswalk builder (rapidfuzz name matching
   against the company master and Exhibit-21 subsidiary lists), with the
   manual override file as final authority. Target and report a mapped-event
   share via `Crosswalk.coverage_report`.
3. Universe membership from listings + market caps, delisting returns
   included. WRDS/CRSP via university access is the preferred source.
4. Feature store build entry point; rebuild determinism test (raw in,
   identical features out).
5. Run the grid. Write the paper, including the decay and capacity section.
   "Where the signal dies and why" is the closing chapter, not a footnote.

## Data sources

* FCC EAS: OET web services (KDB 953436) and opendata.fcc.gov bulk files;
  fccid.io as a cross-check mirror only.
* SEC EDGAR: company master and 10-K Exhibit-21 subsidiary lists (respect
  the fair-access policy; set a real User-Agent in configs/pipeline.yaml).
* Fama-French daily factors: Ken French data library.
* Prices with delisting returns: WRDS/CRSP via USC access preferred.
* Optional corroboration: import bill-of-lading samples (vendor data).
