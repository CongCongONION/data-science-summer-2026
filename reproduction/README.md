# IDX Exchange — CRMLS Analytics (Independent Reproduction)

**Author: Anna** · November 2026

This folder is an independent re-run of the IDX Exchange data-analyst
project, extended with the controls and modelling layer a production
analytics project is expected to carry. The data sources, definitions and
thresholds of the core pipeline follow the team project specification;
the code was written for this reproduction and **every reported number
comes from executing this code** — no result was copied from the original
repository's reports.

## Data sources
- **CRMLS** Sold and Listing records (CoreLogic Trestle API exports),
  monthly files, Jan 2024 – Mar 2026, Residential properties only.
  Raw exports are licensed and are *not* stored in git — place them in
  `data/raw/Sold/` and `data/raw/Listing/`.
- **FRED MORTGAGE30US** — 30-year fixed mortgage rate, weekly, averaged to
  monthly at run time and merged onto both datasets.

## Architecture

| Layer | Module | What it does |
|---|---|---|
| Core pipeline (reproduced method) | `idx_pipeline.py` | 6 stages: aggregate → EDA + rate enrichment → cleaning I → cleaning II (California bounds, dtype audit) → features + segments → business rules + 1.5×IQR outliers |
| Controls | `validate_quality.py` | Row-count reconciliation bridge that must close exactly; data-quality scorecard with named checks and thresholds |
| Modelling (extension) | `hedonic_model.py` | Log-log hedonic OLS with a time-based out-of-sample split, a naive-benchmark comparison, and leakage controls |
| Analytics (extension) | `market_analytics.py` | Monthly price index, mortgage-rate overlay correlations, county deep dive with dispersion, supply-side counts |
| Reporting | `summarize_run.py`, `make_report.py` | `RESULTS.md` and `ANALYSIS_MEMO.md` generated from the run's fact files — never hand-typed |
| Tests | `tests/test_integrity.py` | Pytest suite that re-derives the published numbers from the committed outputs |

The original project specification contains no predictive model; its
method layer is the feature definitions, segment aggregations and the
IQR rule. The hedonic model and market analytics above are labelled
**extensions** — they sit downstream of the reproduced pipeline and do
not alter its definitions.

## Controls & validation (the part reviewers check first)
- **Reconciliation bridge** (`output/row_count_bridge.csv`): every
  ingested row is accounted for — analysis-ready, business-rule invalid,
  or statistical outlier. The bridge is asserted in code; a leak fails
  the run instead of reaching the report.
- **Data-quality scorecard** (`output/data_quality_scorecard.csv`):
  completeness, validity, uniqueness, timeline consistency, geographic
  validity and enrichment checks, each with its measured value,
  threshold and PASS / WARN / FAIL.
- **Labelled populations**: segment/trend tables describe the
  feature-valid population *before* outlier exclusion; the model and
  deep-dive analytics use the analysis-ready population. The two are
  recorded separately in `run_facts.json` and never mixed.
- **Provenance**: every run records `data_kind` in `run_facts.json`; the
  sample generator leaves a marker file, so a sample run can never
  masquerade as a real-data run.
- **Leakage control**: sale-to-list ratios are excluded from the pricing
  model's predictors because they are constructed from the target.

## Modelling summary
Log-log hedonic OLS: `log(ClosePrice)` on `log(LivingArea)`, beds, baths,
lot size, property age, days-on-market, the 30-yr rate, and county /
subtype fixed effects, estimated with numpy (coefficients ship with
standard errors and t-statistics). Validation is a **time-based split**
(train on earlier closes, test on later ones) against a naive
median-$/sqft benchmark scored on the same test rows — see
`output/ANALYSIS_MEMO.md` for this run's out-of-sample metrics and an
honest reading of them.

## Data dictionary (key fields)
| Field | Meaning |
|---|---|
| `ClosePrice` / `ListPrice` / `OriginalListPrice` | Sold price / final list / original list (USD) |
| `LivingArea` | Interior living area, sq ft |
| `ppsf` | `ClosePrice / LivingArea` ($ per sq ft) |
| `price_ratio` | `ClosePrice / OriginalListPrice` (sale-to-list) |
| `DaysOnMarket` | Listing-to-contract marketing time |
| `listing_to_contract_days` / `contract_to_close_days` | Timeline spans from the three contract dates |
| `rate_30yr_fixed` | FRED MORTGAGE30US monthly mean for the close month |
| `exclude_flag` | 1 = removed at the outlier stage (business rule or IQR) |

## Limitations
- Outlier exclusion removes roughly a tenth of rows by design; medians
  after exclusion are not comparable to raw-feed medians without that
  context.
- The Week 5 specification *flags* out-of-California coordinates for
  audit rather than excluding them; the scorecard reports the residual
  share instead of hiding it.
- The hedonic model is a linear benchmark (no spatial effects, classical
  standard errors) — a validation reference point, not a pricing product.
- Thin tail months (late closings of the final listing months) are
  excluded from index/overlay statistics below 10 sales.

## Run it
```bash
pip install -r requirements.txt
python make_sample_data.py     # only if you have no real CRMLS exports yet
python idx_pipeline.py all     # or stage numbers, e.g. python idx_pipeline.py 1 2
python validate_quality.py
python hedonic_model.py
python market_analytics.py
python make_report.py          # output/ANALYSIS_MEMO.md
python summarize_run.py        # output/RESULTS.md
pytest tests/ -q
```
Large intermediate datasets stay under `data/work/` and are not
committed; `output/` carries the small, auditable artifacts.
