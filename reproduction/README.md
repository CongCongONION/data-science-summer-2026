# IDX Exchange — CRMLS Analytics: An Independent Reproduction & Controls Case Study

**Author: Anna** · November 2026 · IDX Exchange Summer 2026 Data Science Internship

## Case summary
The team project specified a six-stage analytics pipeline over CRMLS
real-estate data (Sold + Listing feeds, Jan 2024 – Mar 2026) enriched with
FRED mortgage rates. This folder contains an **independent
re-implementation** of that specification — the code was written from the
spec, and every number reported here was produced by executing it — plus
a **controls and modelling layer** built on top, because a pipeline that
cannot prove its own numbers is not a deliverable.

The work is presented as a case study in the three disciplines that
decide whether data work survives review in a treasury / risk analytics
environment: **pipeline engineering, validation, and control design.**

## What was built

| Layer | Module | Content |
|---|---|---|
| Core pipeline | `idx_pipeline.py` | Aggregate 54 monthly files → EDA + rate enrichment → two-stage cleaning (schema/type audits, timeline flags, geo audit, California bounding box) → features (`ppsf`, sale-to-list, timeline spans) + segments → business rules + 1.5×IQR outliers → **final data-quality gate** |
| Controls | `validate_quality.py` | Row-count reconciliation bridge (asserted, closes exactly); 15-check data-quality scorecard with explicit thresholds |
| Modelling | `hedonic_model.py` | Log-log hedonic OLS (numpy), time-based out-of-sample split, naive-benchmark comparison, target-leakage controls, coefficients with standard errors |
| Analytics | `market_analytics.py` | Monthly price index, mortgage-rate overlay (Pearson + Spearman), county deep dive with dispersion, supply-side counts |
| Reporting | `summarize_run.py` · `make_report.py` | `RESULTS.md` / `ANALYSIS_MEMO.md` generated from the run's fact files — never hand-typed |
| Tests | `tests/test_integrity.py` | 12 pytest checks that re-derive the published numbers from the committed artifacts |

## Three control problems found — and fixed

**1. Rows could disappear without a trace.**
The original design cleans in stages but never reconciles them. Fix: a
**reconciliation bridge** that accounts for every ingested row across
every boundary — this run: 6,549 = 85 business-rule invalids + 734
statistical outliers + 89 gated + **5,641 analysis-ready**. The bridge is
asserted in code; a leak fails the run, not the reviewer.

**2. The cleaning stages flag defects but ship them anyway.**
By spec, suspect records are *flagged for audit* and remain in the
analysis set — 1.5% of "clean" rows carried coordinates outside
California. Flag-and-audit is right *during* cleaning; at the delivery
boundary it is a control failure. Fix (a deliberate, documented deviation
from the spec, isolated in one stage): a **final data-quality gate**
hard-excludes any surviving record carrying a validity flag, each
attributed to one primary reason. Missing coordinates are *not* gated —
absence of information is not an error. The scorecard now reads
**15 PASS / 0 WARN / 0 FAIL**, and a pytest post-condition verifies the
gate on the shipped file itself, not on the code's say-so.

**3. Two different populations were being reported as one.**
Segment tables describe the feature-valid population *before* outlier
exclusion (6,421 rows); the model uses the analysis-ready population
(5,641). Unlabelled, any cross-reference between them is silently wrong.
Fix: both populations are named and recorded in `run_facts.json`, the
distinction is stated wherever the numbers appear, and a test fails if
the segment tables ever drift from their labelled population.

## Modelling, reported honestly
Hedonic OLS on the gated dataset (train 3,314 closes before 2025-07-01,
test 2,416 after): out-of-sample R² **0.644**, RMSE **$248K**, MAPE
**23.4%**; recovered living-area elasticity **0.99** (t ≈ 90). Against a
naive median-$/sqft benchmark scored on the same test rows, the model is
**statistically tied** on RMSE — reported as a tie, not tuned into a
win. The model's value at this stage is driver evidence and a validated
reference specification; sale-to-list ratios are excluded from the
predictors because they are built from the target variable.

## Why this maps to Treasury Analytics
The transferable content is not real-estate trivia; it is the operating
pattern of a treasury/risk data function: multi-source position-style
feeds ingested on a fixed calendar, an external market series (FRED
30-yr) merged as the rate overlay, two-stage data-quality controls with
an asserted reconciliation, a hard gate before anything reaches a
report, model validation with an out-of-sample design and a benchmark,
and an audit trail in which every published figure can be re-derived
from committed artifacts by an independent test suite.

## Data & provenance
- **CRMLS** Sold/Listing exports (CoreLogic Trestle), Residential only;
  licensed raw files are not stored in git — place them in
  `data/raw/Sold/` and `data/raw/Listing/` and re-run.
- **FRED MORTGAGE30US**, weekly, averaged to monthly at run time.
- This execution ran on **synthetic sample data in the CRMLS schema**
  (the real exports are not in any repository); the mortgage series is
  real. Provenance is recorded per run (`data_kind` in `run_facts.json`,
  backed by a generator marker file), so a sample run can never
  masquerade as a real-data run. All controls are data-agnostic.

## Limitations
- Outlier + gate exclusions remove ~14% of ingested rows by design;
  post-exclusion medians are not comparable to raw-feed medians.
- The hedonic model is a linear benchmark (no spatial effects, classical
  standard errors) — a validation reference, not a pricing product.
- Thin tail months (< 10 sales) are excluded from index/overlay stats.

## Run it
```bash
pip install -r requirements.txt
python make_sample_data.py     # only if you have no real CRMLS exports yet
python idx_pipeline.py all
python validate_quality.py
python hedonic_model.py
python market_analytics.py
python make_report.py          # output/ANALYSIS_MEMO.md
python summarize_run.py        # output/RESULTS.md
pytest tests/ -q               # 12 checks over the committed artifacts
```

## Key fields
| Field | Meaning |
|---|---|
| `ClosePrice` / `ListPrice` / `OriginalListPrice` | Sold / final list / original list price (USD) |
| `ppsf` · `price_ratio` | `ClosePrice / LivingArea` · `ClosePrice / OriginalListPrice` |
| `listing_to_contract_days` · `contract_to_close_days` | Timeline spans from the three contract dates |
| `rate_30yr_fixed` | FRED monthly mean for the close month |
| `gate_flag` · `exclude_flag` | Removed at the final DQ gate · removed at any exclusion layer |
