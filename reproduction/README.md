# IDX Exchange — CRMLS Analytics (Independent Reproduction)

**Author: Anna** · November 2026

This folder is an independent re-run of the IDX Exchange data-analyst
project. The data sources, definitions and thresholds follow the team
project specification; the code was written for this reproduction and
**every reported number comes from executing this code** — no result was
copied from the original repository's reports.

## Data sources
- **CRMLS** Sold and Listing records (CoreLogic Trestle API exports),
  monthly files, Jan 2024 – Mar 2026, Residential properties only.
  Raw exports are licensed and are *not* stored in git — place them in
  `data/raw/Sold/` and `data/raw/Listing/`.
- **FRED MORTGAGE30US** — 30-year fixed mortgage rate, weekly, averaged to
  monthly at run time and merged onto both datasets.

## Method (what is being reproduced)
1. Aggregate monthly files → combined Sold / Listing (Residential only)
2. Missing-value profile (>90% = sparse), distributions for 9 numeric
   fields, mortgage-rate enrichment
3. Cleaning I: drop sparse columns, standardise dates, flag timeline
   contradictions, invalid prices / areas / days-on-market, geo audit
4. Cleaning II: California bounding-box check (lat 32.5–42.0,
   lon −124.4…−114.1), dtype audit, final flag roll-up
5. Features: `price_ratio`, `ppsf`, calendar parts,
   `listing_to_contract_days`, `contract_to_close_days`; segments by
   county / subtype / listing office; monthly market trend
6. Outliers: business-rule invalids, then 1.5×IQR flags on six fields →
   `sold_analysis_ready.csv`

There is no predictive machine-learning model in this project; the method
layer is the feature definitions, segment aggregations and the IQR rule
above, and this reproduction keeps it that way.

## Run it
```bash
pip install -r requirements.txt
python make_sample_data.py     # only if you have no real CRMLS exports yet
python idx_pipeline.py all     # or stage numbers, e.g. python idx_pipeline.py 1 2
python summarize_run.py        # regenerate output/RESULTS.md from the run record
```
Each run writes its own facts to `output/run_facts.json`, including a
`data_kind` provenance field (the sample generator leaves a marker file, so
a sample run can never masquerade as a real-data run). `output/RESULTS.md`
is generated from that record by `summarize_run.py`, never typed by hand —
replace the raw files with the real exports and re-run for production
numbers. Large intermediate datasets stay under `data/work/` and are not
committed.
