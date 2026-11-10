"""
week2_eda_mortgage.py
Weeks 2-3 Deliverable: Dataset Structuring, Validation, and Mortgage Rate Enrichment

Covers:
  1. Dataset Understanding  — shape, dtypes, high-missing columns
  2. Missing Value Analysis — null counts/pcts, flag >90% null columns
  3. Numeric Distribution   — histograms, boxplots, percentile summaries for
                              ClosePrice, LivingArea, DaysOnMarket (sold dataset)
  4. Suggested EDA Questions answered for sold dataset
  5. Mortgage Rate Enrichment — fetch FRED MORTGAGE30US, resample weekly→monthly,
                                merge onto both sold and listings, validate
  6. Save outputs:
       - sold_residential_eda.csv        (filtered sold, no mortgage yet)
       - listed_residential_eda.csv      (filtered listed, no mortgage yet)
       - sold_with_rates.csv             (sold + monthly mortgage rate)
       - listed_with_rates.csv           (listed + monthly mortgage rate)

Author: Ruoyu Wang
Date:   April 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# 0. PATHS
# ============================================================

WEEK1_DIR  = r"D:\IDXexchange\week1"
WEEK2_DIR  = r"D:\IDXexchange\week2"

SOLD_FILE   = os.path.join(WEEK1_DIR, "combined_sold.csv")
LISTED_FILE = os.path.join(WEEK1_DIR, "combined_listed.csv")

PLOT_DIR = os.path.join(WEEK2_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING COMBINED DATASETS")
print("=" * 70)

sold   = pd.read_csv(SOLD_FILE,   low_memory=False, encoding='latin-1')
listed = pd.read_csv(LISTED_FILE, low_memory=False, encoding='latin-1')

print(f"  Sold   — rows before filter: {len(sold):,}   | columns: {sold.shape[1]}")
print(f"  Listed — rows before filter: {len(listed):,} | columns: {listed.shape[1]}")

# ============================================================
# 2. DATASET UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: DATASET UNDERSTANDING")
print("=" * 70)

for label, df in [("SOLD", sold), ("LISTED", listed)]:
    print(f"\n  --- {label} dataset ---")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\n  Column data types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, cnt in dtype_counts.items():
        print(f"    {str(dtype):<15} : {cnt} columns")

# ============================================================
# 3. UNIQUE PROPERTY TYPES & RESIDENTIAL FILTER
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: PROPERTY TYPES & RESIDENTIAL FILTER")
print("=" * 70)

for label, df in [("SOLD", sold), ("LISTED", listed)]:
    print(f"\n  --- {label} ---")
    if 'PropertyType' in df.columns:
        pt_counts = df['PropertyType'].value_counts(dropna=False)
        print(f"  Unique PropertyType values found:")
        for pt, cnt in pt_counts.items():
            pct = cnt / len(df) * 100
            print(f"    '{pt}' : {cnt:,} rows ({pct:.1f}%)")
    else:
        print("  [WARNING] 'PropertyType' column not found")

# Apply filter — data from week1 should already be Residential only,
# but we re-apply here to be explicit and document the logic.
sold_before   = len(sold)
listed_before = len(listed)

sold   = sold[sold['PropertyType']   == 'Residential'].copy()
listed = listed[listed['PropertyType'] == 'Residential'].copy()

sold.reset_index(drop=True, inplace=True)
listed.reset_index(drop=True, inplace=True)

print(f"\n  Filter applied: PropertyType == 'Residential'")
print(f"  Sold   : {sold_before:,} → {len(sold):,} rows  (dropped {sold_before - len(sold):,})")
print(f"  Listed : {listed_before:,} → {len(listed):,} rows  (dropped {listed_before - len(listed):,})")

# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: MISSING VALUE ANALYSIS")
print("=" * 70)

def missing_report(df, label):
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        'null_count': null_counts,
        'null_pct':   null_pct
    }).sort_values('null_pct', ascending=False)

    print(f"\n  === {label} — Full null-count table ===")
    print(f"  {'Column':<40} {'Null Count':>12} {'Null %':>8}")
    print("  " + "-" * 62)
    for col, row in summary.iterrows():
        flag = " *** >90%" if row['null_pct'] > 90 else ""
        print(f"  {col:<40} {row['null_count']:>12,.0f} {row['null_pct']:>7.1f}%{flag}")

    high_null = summary[summary['null_pct'] > 90]
    print(f"\n  Columns with >90% missing ({len(high_null)} total):")
    if len(high_null) == 0:
        print("    None — no columns exceed 90% null threshold.")
    else:
        for col, row in high_null.iterrows():
            print(f"    DROP CANDIDATE: {col} — {row['null_pct']}% null")

    return summary

sold_null_summary   = missing_report(sold,   "SOLD")
listed_null_summary = missing_report(listed, "LISTED")

# Save null summaries as CSV
sold_null_summary.to_csv(os.path.join(WEEK2_DIR, "sold_null_summary.csv"))
listed_null_summary.to_csv(os.path.join(WEEK2_DIR, "listed_null_summary.csv"))
print(f"\n  Saved: sold_null_summary.csv  &  listed_null_summary.csv")

# ============================================================
# 5. NUMERIC DISTRIBUTION REVIEW (SOLD dataset)
#    Fields: ClosePrice, ListPrice, OriginalListPrice, LivingArea,
#            LotSizeAcres, BedroomsTotal, BathroomsTotalInteger,
#            DaysOnMarket, YearBuilt
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: NUMERIC DISTRIBUTION REVIEW (SOLD dataset)")
print("=" * 70)

NUMERIC_FIELDS = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt'
]

percentile_rows = []

for field in NUMERIC_FIELDS:
    if field not in sold.columns:
        print(f"\n  [SKIP] {field} — column not found")
        continue

    s = pd.to_numeric(sold[field], errors='coerce')
    non_null = s.dropna()

    if len(non_null) == 0:
        print(f"\n  [SKIP] {field} — all values are null after coercion")
        continue

    p = {
        'field':   field,
        'count':   len(non_null),
        'null':    s.isna().sum(),
        'min':     non_null.min(),
        'p5':      non_null.quantile(0.05),
        'p25':     non_null.quantile(0.25),
        'median':  non_null.median(),
        'mean':    non_null.mean(),
        'p75':     non_null.quantile(0.75),
        'p95':     non_null.quantile(0.95),
        'max':     non_null.max(),
        'std':     non_null.std(),
    }
    percentile_rows.append(p)

    print(f"\n  {field}:")
    print(f"    Count  : {p['count']:,}   |  Null: {p['null']:,}")
    print(f"    Min    : {p['min']:,.2f}")
    print(f"    p5     : {p['p5']:,.2f}")
    print(f"    p25    : {p['p25']:,.2f}")
    print(f"    Median : {p['median']:,.2f}")
    print(f"    Mean   : {p['mean']:,.2f}")
    print(f"    p75    : {p['p75']:,.2f}")
    print(f"    p95    : {p['p95']:,.2f}")
    print(f"    Max    : {p['max']:,.2f}")
    print(f"    Std    : {p['std']:,.2f}")

    # Outlier flags
    if field in ('ClosePrice', 'ListPrice', 'OriginalListPrice'):
        bad = (s <= 0).sum()
        if bad:
            print(f"    [OUTLIER] {bad:,} records with {field} <= 0")
        extreme = (s > 50_000_000).sum()
        if extreme:
            print(f"    [OUTLIER] {extreme:,} records with {field} > $50M")
    if field == 'DaysOnMarket':
        bad = (s < 0).sum()
        if bad:
            print(f"    [OUTLIER] {bad:,} records with negative DaysOnMarket")
    if field == 'LivingArea':
        bad = (s <= 0).sum()
        if bad:
            print(f"    [OUTLIER] {bad:,} records with LivingArea <= 0")

    # --- Histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Sold — {field}", fontsize=13, fontweight='bold')

    # Clip to p95 for readability
    clip_val = non_null.quantile(0.99)
    clipped  = non_null.clip(upper=clip_val)

    axes[0].hist(clipped, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    axes[0].set_title("Histogram (clipped at p99)")
    axes[0].set_xlabel(field)
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'
    ))

    axes[1].boxplot(clipped, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title("Boxplot (clipped at p99)")
    axes[1].set_xlabel(field)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'
    ))

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"dist_{field}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Saved plot: plots/dist_{field}.png")

# Save percentile summary
dist_df = pd.DataFrame(percentile_rows).set_index('field')
dist_df.to_csv(os.path.join(WEEK2_DIR, "sold_numeric_distribution.csv"))
print(f"\n  Saved: sold_numeric_distribution.csv")

# ============================================================
# 6. SUGGESTED EDA QUESTIONS (SOLD dataset)
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: SUGGESTED EDA QUESTIONS (SOLD dataset)")
print("=" * 70)

# Q1: Residential vs other property type share
# (already shown in Step 3)
print("\n  Q1: Residential vs other property type share")
print("      → See Step 3 above. After filter: 100% Residential.")

# Q2: Median and average close prices
if 'ClosePrice' in sold.columns:
    cp = pd.to_numeric(sold['ClosePrice'], errors='coerce').dropna()
    print(f"\n  Q2: Close price summary")
    print(f"      Median : ${cp.median():,.0f}")
    print(f"      Mean   : ${cp.mean():,.0f}")

# Q3: Days on Market distribution
if 'DaysOnMarket' in sold.columns:
    dom = pd.to_numeric(sold['DaysOnMarket'], errors='coerce').dropna()
    print(f"\n  Q3: Days on Market distribution")
    print(f"      Median : {dom.median():.0f} days")
    print(f"      Mean   : {dom.mean():.1f} days")
    print(f"      p75    : {dom.quantile(0.75):.0f} days")
    print(f"      p90    : {dom.quantile(0.90):.0f} days")
    print(f"      Max    : {dom.max():.0f} days")

# Q4: Pct sold above vs below list price
if 'ClosePrice' in sold.columns and 'ListPrice' in sold.columns:
    cp  = pd.to_numeric(sold['ClosePrice'],  errors='coerce')
    lp  = pd.to_numeric(sold['ListPrice'],   errors='coerce')
    both = cp.notna() & lp.notna() & (lp > 0)
    above = ((cp[both] >  lp[both]).sum())
    at    = ((cp[both] == lp[both]).sum())
    below = ((cp[both] <  lp[both]).sum())
    total = both.sum()
    print(f"\n  Q4: Sold above / at / below list price")
    print(f"      Above : {above:,} ({above/total*100:.1f}%)")
    print(f"      At    : {at:,}    ({at/total*100:.1f}%)")
    print(f"      Below : {below:,} ({below/total*100:.1f}%)")

# Q5: Date consistency issues
if all(c in sold.columns for c in ['ListingContractDate', 'CloseDate']):
    ld = pd.to_datetime(sold['ListingContractDate'], errors='coerce')
    cd = pd.to_datetime(sold['CloseDate'],           errors='coerce')
    bad = (ld > cd).sum()
    print(f"\n  Q5: Date consistency — ListingContractDate > CloseDate")
    print(f"      Records with listing date AFTER close date: {bad:,}")

# Q6: Top counties by median price
if 'CountyOrParish' in sold.columns and 'ClosePrice' in sold.columns:
    cp_col = pd.to_numeric(sold['ClosePrice'], errors='coerce')
    county_med = (sold.assign(ClosePrice_num=cp_col)
                      .groupby('CountyOrParish')['ClosePrice_num']
                      .agg(['median', 'count'])
                      .rename(columns={'median': 'MedianPrice', 'count': 'Transactions'})
                      .query('Transactions >= 50')
                      .sort_values('MedianPrice', ascending=False)
                      .head(10))
    print(f"\n  Q6: Top 10 counties by median close price (min 50 transactions)")
    print(f"  {'County':<30} {'Median Price':>14} {'Transactions':>14}")
    print("  " + "-" * 60)
    for county, row in county_med.iterrows():
        print(f"  {str(county):<30} ${row['MedianPrice']:>13,.0f} {row['Transactions']:>14,.0f}")

# ============================================================
# 7. SAVE FILTERED DATASETS (before mortgage enrichment)
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: SAVING FILTERED DATASETS")
print("=" * 70)

sold_eda_path   = os.path.join(WEEK2_DIR, "sold_residential_eda.csv")
listed_eda_path = os.path.join(WEEK2_DIR, "listed_residential_eda.csv")

sold.to_csv(sold_eda_path,     index=False, encoding='utf-8-sig')
listed.to_csv(listed_eda_path, index=False, encoding='utf-8-sig')

print(f"  Saved: sold_residential_eda.csv   ({len(sold):,} rows)")
print(f"  Saved: listed_residential_eda.csv ({len(listed):,} rows)")

# ============================================================
# 8. MORTGAGE RATE ENRICHMENT (FRED MORTGAGE30US)
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: MORTGAGE RATE ENRICHMENT")
print("=" * 70)

# --- Step 8a: Fetch weekly MORTGAGE30US from FRED ---
print("\n  8a. Fetching MORTGAGE30US from FRED ...")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

mortgage = pd.read_csv(FRED_URL)
# FRED column names: 'observation_date', 'MORTGAGE30US'
mortgage.columns = ['date', 'rate_30yr_fixed']
mortgage['date'] = pd.to_datetime(mortgage['date'])

# FRED encodes missing weeks as '.' — coerce to NaN and drop
mortgage['rate_30yr_fixed'] = pd.to_numeric(mortgage['rate_30yr_fixed'], errors='coerce')
mortgage.dropna(subset=['rate_30yr_fixed'], inplace=True)

print(f"  Fetched {len(mortgage):,} weekly observations")
print(f"  Date range: {mortgage['date'].min().date()} → {mortgage['date'].max().date()}")
print(f"  Rate range: {mortgage['rate_30yr_fixed'].min():.2f}% → {mortgage['rate_30yr_fixed'].max():.2f}%")

# --- Step 8b: Resample weekly → monthly average ---
print("\n  8b. Resampling to monthly averages ...")
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)
mortgage_monthly['rate_30yr_fixed'] = mortgage_monthly['rate_30yr_fixed'].round(4)

print(f"  Monthly observations: {len(mortgage_monthly)}")
print(f"\n  Sample of monthly rates (most recent 6 months):")
print(f"  {'Year-Month':<12} {'Rate (%)':>10}")
print("  " + "-" * 25)
for _, row in mortgage_monthly.tail(6).iterrows():
    print(f"  {str(row['year_month']):<12} {row['rate_30yr_fixed']:>9.2f}%")

# Save mortgage monthly table
mortgage_monthly.to_csv(
    os.path.join(WEEK2_DIR, "mortgage_monthly_avg.csv"), index=False
)
print(f"\n  Saved: mortgage_monthly_avg.csv")

# --- Step 8c: Create year_month key on MLS datasets ---
print("\n  8c. Creating year_month join keys ...")

# Sold — key off CloseDate
sold['CloseDate_dt']         = pd.to_datetime(sold['CloseDate'], errors='coerce')
sold['year_month']           = sold['CloseDate_dt'].dt.to_period('M')

# Listed — key off ListingContractDate
listed['ListingContractDate_dt'] = pd.to_datetime(listed['ListingContractDate'], errors='coerce')
listed['year_month']             = listed['ListingContractDate_dt'].dt.to_period('M')

sold_nm   = sold['year_month'].isna().sum()
listed_nm = listed['year_month'].isna().sum()
print(f"  Sold   — null year_month keys: {sold_nm:,}")
print(f"  Listed — null year_month keys: {listed_nm:,}")

# --- Step 8d: Merge ---
print("\n  8d. Merging mortgage rates ...")
sold_with_rates   = sold.merge(mortgage_monthly, on='year_month', how='left')
listed_with_rates = listed.merge(mortgage_monthly, on='year_month', how='left')

print(f"  Sold   row count after merge: {len(sold_with_rates):,} (expected {len(sold):,})")
print(f"  Listed row count after merge: {len(listed_with_rates):,} (expected {len(listed):,})")

# --- Step 8e: Validate ---
print("\n  8e. Validation — checking for null mortgage rates after merge ...")

sold_null_rate   = sold_with_rates['rate_30yr_fixed'].isna().sum()
listed_null_rate = listed_with_rates['rate_30yr_fixed'].isna().sum()

if sold_null_rate == 0:
    print("  [PASS] Sold   — no null rate_30yr_fixed values after merge")
else:
    print(f"  [WARNING] Sold   — {sold_null_rate:,} null rate values")
    # Show which months couldn't be matched
    unmatched = (sold_with_rates[sold_with_rates['rate_30yr_fixed'].isna()]
                 ['year_month'].value_counts())
    print("    Unmatched year_month values:")
    for ym, cnt in unmatched.items():
        print(f"      {ym} : {cnt:,} records")

if listed_null_rate == 0:
    print("  [PASS] Listed — no null rate_30yr_fixed values after merge")
else:
    print(f"  [WARNING] Listed — {listed_null_rate:,} null rate values")
    unmatched = (listed_with_rates[listed_with_rates['rate_30yr_fixed'].isna()]
                 ['year_month'].value_counts())
    print("    Unmatched year_month values:")
    for ym, cnt in unmatched.items():
        print(f"      {ym} : {cnt:,} records")

# Preview
print("\n  Preview — Sold with rates (first 5 rows):")
preview_cols = ['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']
preview_cols = [c for c in preview_cols if c in sold_with_rates.columns]
print(sold_with_rates[preview_cols].head().to_string(index=False))

# ============================================================
# 9. SAVE ENRICHED DATASETS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: SAVING ENRICHED DATASETS")
print("=" * 70)

# Convert Period column to string so CSV serialization works
sold_with_rates['year_month']   = sold_with_rates['year_month'].astype(str)
listed_with_rates['year_month'] = listed_with_rates['year_month'].astype(str)

# Drop the temporary _dt columns (redundant)
sold_with_rates.drop(columns=['CloseDate_dt'], errors='ignore', inplace=True)
listed_with_rates.drop(columns=['ListingContractDate_dt'], errors='ignore', inplace=True)

sold_rates_path   = os.path.join(WEEK2_DIR, "sold_with_rates.csv")
listed_rates_path = os.path.join(WEEK2_DIR, "listed_with_rates.csv")

sold_with_rates.to_csv(sold_rates_path,     index=False, encoding='utf-8-sig')
listed_with_rates.to_csv(listed_rates_path, index=False, encoding='utf-8-sig')

print(f"  Saved: sold_with_rates.csv    ({len(sold_with_rates):,} rows)")
print(f"  Saved: listed_with_rates.csv  ({len(listed_with_rates):,} rows)")

# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "#" * 70)
print("  WEEKS 2-3 DELIVERABLE — COMPLETE")
print("#" * 70)

print(f"""
  Output files (all in {WEEK2_DIR}):
    sold_null_summary.csv          — null counts/pcts for every column (sold)
    listed_null_summary.csv        — null counts/pcts for every column (listed)
    sold_numeric_distribution.csv  — percentile summary for 9 numeric fields
    sold_residential_eda.csv       — filtered sold dataset (Residential only)
    listed_residential_eda.csv     — filtered listed dataset (Residential only)
    mortgage_monthly_avg.csv       — FRED MORTGAGE30US resampled to monthly avg
    sold_with_rates.csv            — sold dataset enriched with mortgage rate
    listed_with_rates.csv          — listed dataset enriched with mortgage rate
    plots/dist_<field>.png         — histogram + boxplot for each numeric field
""")
