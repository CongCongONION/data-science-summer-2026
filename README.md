# IDX Exchange — Data Analyst Internship

Python-based MLS analytics pipeline using CRMLS data. The project progresses from raw data extraction through cleaning, feature engineering, and Tableau dashboard development over 12 weeks.

---

## Project Structure

```
IDXexchange/
├── raw/                          # Raw monthly CSV files from CoreLogic Trestle API
│   ├── CRMLSListing202401.csv
│   ├── CRMLSSold202401.csv
│   └── ...                       # Jan 2024 – Mar 2026
│
├── week1/                        # Dataset aggregation
│   ├── crmls_sold.py             # API extraction script — sold transactions
│   ├── crmls_listed.py           # API extraction script — listings
│   ├── sold_analysis.py          # ETL pipeline for sold transactions
│   ├── listed_analysis.py        # ETL pipeline for listed properties
│   ├── check_combined.py         # Post-aggregation validation
│   ├── combined_sold.csv         # All sold records, Jan 2024 – Mar 2026
│   └── combined_listed.csv       # All listing records, Jan 2024 – Mar 2026
│
├── week2-3/                      # EDA, validation & mortgage enrichment
│   ├── eda_mortgage.py           # Main analysis script
│   ├── plots/                    # Distribution plots for 9 numeric fields
│   ├── data/                     # EDA outputs and merged datasets
│   └── report/                   # week2_report.html
│
├── week4/                        # Data cleaning — phase 1
│   ├── data_cleaning.py          # Null column drops, date parsing, numeric flags
│   ├── generate_report.py        # HTML report generator
│   ├── data/                     # sold_clean_flagged.csv, listed_clean_flagged.csv
│   └── report/                   # week3_report.html, week4_5_report.html
│
├── week5/                        # Data cleaning — phase 2 (coordinate validation)
│   ├── data_cleaning_wk5.py      # California bounds check, dtype audit
│   ├── generate_report.py
│   ├── data/                     # sold_final_clean.csv, listed_final_clean.csv
│   └── report/                   # week5_report.html
│
├── week6/                        # Feature engineering & segment analysis
│   ├── feature_engineering.py    # Engineered features + county/subtype/office segments
│   ├── generate_report.py
│   ├── data/                     # sold_with_features.csv, segment CSVs, monthly trend
│   └── report/                   # week6_report.html
│
├── week7/                        # Outlier detection & analysis-ready dataset
│   ├── outlier_detection.py      # IQR method + business-rule flags
│   ├── generate_report.py
│   ├── data/                     # sold_outlier_flagged.csv, sold_analysis_ready.csv
│   └── report/                   # week7_report.html
│
└── README.md
```

---

## Data Pipeline

```
CoreLogic Trestle API → Monthly CSVs → Aggregation → EDA & Cleaning → Feature Engineering → Outlier Detection → Tableau
```

**Two core datasets:**

| Dataset | Description |
|---|---|
| **Listings** | Market supply — all properties that entered the market, including active and failed-to-sell |
| **Sold** | Market outcomes — properties that successfully closed escrow |

Both datasets cover **January 2024 – March 2026**, filtered to **Residential** property type only.

---

## Weekly Progress

### Week 0 — MLS Data Pipeline Orientation
- Reviewed CoreLogic Trestle API authentication and pagination
- Ran `crmls_sold.py` and `crmls_listed.py` to extract monthly CSV files

### Week 1 — Dataset Aggregation
- Concatenated all monthly files (Jan 2024 – Mar 2026) into two combined datasets
- Filtered both to `PropertyType == 'Residential'`
- Validated row counts, date ranges, and duplicate records
- **Output:** `combined_sold.csv` (414,184 rows), `combined_listed.csv` (566,673 rows)

### Weeks 2–3 — Dataset Structuring, Validation & Mortgage Enrichment
- Documented unique property types and filtering logic
- Full missing value analysis — flagged 17 columns (sold) and 13 columns (listed) with >90% null
- Numeric distribution review for 9 key fields: histograms, boxplots, percentile summaries
- Answered core EDA questions: median close price ($822K), DOM distribution, above/below list price split, date consistency errors
- Fetched FRED `MORTGAGE30US` weekly series, resampled to monthly averages, merged onto both datasets
- **Output:** `sold_with_rates.csv`, `listed_with_rates.csv`, `week2_report.html`

### Week 4 — Data Cleaning Phase 1
- Dropped 15 high-null columns (Sold) and 13 high-null columns (Listed)
- Parsed and standardized `ListingContractDate`, `PurchaseContractDate`, `CloseDate`
- Flagged timeline anomalies: 61 listing-after-close, 240 purchase-after-close, 270 negative-timeline records
- Added numeric quality flags: invalid prices, LivingArea ≤ 0, negative DaysOnMarket
- Geographic audit: Lat/Lon 3.85% null (sold), 14.2% null (listed); PostalCode 100% complete
- **Output:** `sold_clean_flagged.csv` (414,184 × 83), `listed_clean_flagged.csv` (566,673 × 79), `week4_5_report.html`

### Week 5 — Data Cleaning Phase 2 (Coordinate Validation)
- Applied California geographic bounding box to flag implausible coordinates: 65 sold, 266 listed
- Full data type audit across all numeric and date fields
- Compiled final flag summary: 819 any-invalid-flag (sold, 0.20%), 531 (listed, 0.09%)
- **Output:** `sold_final_clean.csv` (414,184 × 84), `listed_final_clean.csv` (566,673 × 80), `week5_report.html`

### Week 6 — Feature Engineering
- Engineered 8 analytical features:
  - `price_ratio` / `close_to_orig_list_ratio` — ClosePrice / OriginalListPrice (99.82% valid)
  - `ppsf` — price per sq ft (99.91% valid)
  - `close_year`, `close_month`, `yr_mo` — temporal decomposition from CloseDate
  - `listing_to_contract_days` — days from listing to accepted offer (99.95% valid)
  - `contract_to_close_days` — days from offer to escrow close (99.95% valid)
- Segment analysis: 55 counties, 20 property subtypes, top-20 listing offices
- Monthly market trend series: 28 periods (Jan 2024 – Mar 2026)
- **Output:** `sold_with_features.csv` (414,184 × 96), `segment_by_county.csv`, `segment_by_subtype.csv`, `segment_by_office.csv`, `monthly_market_trend.csv`, `week6_report.html`

### Week 7 — Outlier Detection
- Applied IQR method (1.5× IQR) to 6 fields: ClosePrice, LivingArea, DaysOnMarket, ppsf, listing_to_contract_days, contract_to_close_days
- Business rules: flagged 3 invalid ClosePrice, 384 invalid LivingArea, 47 invalid DaysOnMarket records
- Removed 96,966 rows (23.4%) as outliers; retained 317,218 analysis-ready records
- Median ClosePrice shift: $823K → $789K after outlier removal
- **Output:** `sold_outlier_flagged.csv` (full dataset with flags), `sold_analysis_ready.csv` (317,218 rows), `week7_report.html`

### Weeks 8–12 — Tableau Dashboards & Market Report *(upcoming)*
- Tableau dashboard development: market trends, competitive analysis, geographic heatmaps
- Final market report summarizing 2024–2026 CRMLS residential market

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python / Pandas** | Data extraction, cleaning, feature engineering |
| **FRED API** | 30-year fixed mortgage rate enrichment |
| **Matplotlib** | Distribution plots and EDA visualizations |
| **Tableau Public** | Market and competitive analysis dashboards |
| **GitHub** | Version control and collaboration |

---

## Final Deliverables

- `sold_analysis_ready.csv` — 317,218 clean, outlier-filtered, feature-enriched sold transactions
- `listed_final_clean.csv` — clean, analysis-ready listings dataset
- Tableau dashboards published to Tableau Public

---

## Author

**Ruoyu Wang** — Data Analyst Intern @ IDX Exchange
