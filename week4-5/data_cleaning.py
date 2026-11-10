"""
data_cleaning.py
Weeks 4-5 Deliverable: Data Cleaning — Drop Columns, Fix Dates, Remove Invalid Numerics

Covers:
  1. Load enriched datasets from Week 2/3
  2. Column pruning  — drop all columns with >90% null
  3. Date standardization — parse date fields, flag ListingDate > CloseDate
  4. Numeric cleaning  — flag/remove invalid prices, living area, DOM
  5. Geographic check  — null lat/lon summary
  6. Save cleaned outputs + cleaning summary CSVs

Author: Ruoyu Wang
Date:   April 2026
"""

import os
import json
import pandas as pd
import numpy as np

WEEK2_DIR = r"D:\IDXexchange\week2-3\data"
WEEK3_DIR = r"D:\IDXexchange\week4"
DATA_DIR  = os.path.join(WEEK3_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. LOAD
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING WEEK-2 ENRICHED DATASETS")
print("=" * 70)

sold   = pd.read_csv(os.path.join(WEEK2_DIR, "sold_with_rates.csv"),   low_memory=False)
listed = pd.read_csv(os.path.join(WEEK2_DIR, "listed_with_rates.csv"), low_memory=False)

sold_raw_rows   = len(sold)
sold_raw_cols   = sold.shape[1]
listed_raw_rows = len(listed)
listed_raw_cols = listed.shape[1]

print(f"  Sold   — {sold_raw_rows:,} rows × {sold_raw_cols} columns")
print(f"  Listed — {listed_raw_rows:,} rows × {listed_raw_cols} columns")

# ============================================================
# 2. COLUMN PRUNING  (drop >90% null)
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: COLUMN PRUNING — DROP >90% NULL")
print("=" * 70)

def drop_high_null(df, label, threshold=0.90):
    null_pct = df.isnull().mean()
    drop_cols = null_pct[null_pct > threshold].index.tolist()
    kept_cols = [c for c in df.columns if c not in drop_cols]
    df_clean  = df[kept_cols].copy()
    print(f"\n  {label}: dropping {len(drop_cols)} columns (>{threshold*100:.0f}% null)")
    for c in drop_cols:
        print(f"    DROP  {c:40s}  ({null_pct[c]*100:.1f}% null)")
    print(f"  Columns remaining: {df_clean.shape[1]}")
    return df_clean, drop_cols

sold,   sold_dropped_cols   = drop_high_null(sold,   "SOLD")
listed, listed_dropped_cols = drop_high_null(listed, "LISTED")

# ============================================================
# 3. DATE STANDARDISATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: DATE FIELD STANDARDISATION")
print("=" * 70)

DATE_FIELDS_SOLD   = ['ListingContractDate', 'CloseDate',
                      'StatusChangeTimestamp', 'ModificationTimestamp']
DATE_FIELDS_LISTED = ['ListingContractDate', 'ExpirationDate',
                      'StatusChangeTimestamp', 'ModificationTimestamp']

def parse_dates(df, fields, label):
    results = {}
    for col in fields:
        if col not in df.columns:
            print(f"  [{label}] {col}: NOT FOUND — skip")
            continue
        before_null = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        after_null  = df[col].isna().sum()
        new_nulls   = after_null - before_null
        print(f"  [{label}] {col}: parsed — new NaT introduced: {new_nulls:,}")
        results[col] = {'before_null': int(before_null), 'after_null': int(after_null)}
    return df, results

sold,   sold_date_results   = parse_dates(sold,   DATE_FIELDS_SOLD,   "SOLD")
listed, listed_date_results = parse_dates(listed, DATE_FIELDS_LISTED, "LISTED")

# Flag 1: ListingContractDate > CloseDate
flag_col = 'listing_after_close_flag'
if 'ListingContractDate' in sold.columns and 'CloseDate' in sold.columns:
    sold[flag_col] = (
        sold['ListingContractDate'].notna() &
        sold['CloseDate'].notna() &
        (sold['ListingContractDate'] > sold['CloseDate'])
    ).astype(int)
    n_flagged = int(sold[flag_col].sum())
    print(f"\n  Flagged {n_flagged:,} sold records where ListingContractDate > CloseDate")
else:
    n_flagged = 0
    sold[flag_col] = 0

# Flag 2: PurchaseContractDate > CloseDate
flag_purchase = 'purchase_after_close_flag'
if 'PurchaseContractDate' in sold.columns and 'CloseDate' in sold.columns:
    sold[flag_purchase] = (
        sold['PurchaseContractDate'].notna() &
        sold['CloseDate'].notna() &
        (sold['PurchaseContractDate'] > sold['CloseDate'])
    ).astype(int)
    n_purchase_flag = int(sold[flag_purchase].sum())
    print(f"  Flagged {n_purchase_flag:,} sold records where PurchaseContractDate > CloseDate")
else:
    n_purchase_flag = 0
    sold[flag_purchase] = 0

# Flag 3: ListingContractDate > PurchaseContractDate (negative timeline)
flag_neg = 'negative_timeline_flag'
if all(c in sold.columns for c in ['ListingContractDate', 'PurchaseContractDate']):
    sold[flag_neg] = (
        sold['ListingContractDate'].notna() &
        sold['PurchaseContractDate'].notna() &
        (sold['ListingContractDate'] > sold['PurchaseContractDate'])
    ).astype(int)
    n_neg_flag = int(sold[flag_neg].sum())
    print(f"  Flagged {n_neg_flag:,} sold records where ListingContractDate > PurchaseContractDate")
else:
    n_neg_flag = 0
    sold[flag_neg] = 0

# ============================================================
# 4. NUMERIC CLEANING
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: NUMERIC CLEANING — FLAG INVALID VALUES")
print("=" * 70)

cleaning_log = []

def flag_invalid(df, col, label, condition_desc, mask_fn):
    if col not in df.columns:
        print(f"  [{label}] {col}: NOT FOUND — skip")
        return df, 0
    s = pd.to_numeric(df[col], errors='coerce')
    mask = mask_fn(s)
    count = int(mask.sum())
    flag  = f"{col}_invalid_flag"
    df[flag] = mask.astype(int)
    print(f"  [{label}] {col} — {condition_desc}: {count:,} records flagged")
    cleaning_log.append({
        'dataset': label, 'column': col,
        'rule': condition_desc, 'flagged': count
    })
    return df, count

sold, _   = flag_invalid(sold, 'ClosePrice', 'SOLD',
                          'ClosePrice <= 0',   lambda s: s <= 0)
sold, _   = flag_invalid(sold, 'ClosePrice', 'SOLD',
                          'ClosePrice > $50M', lambda s: s > 50_000_000)
sold, _   = flag_invalid(sold, 'ListPrice',  'SOLD',
                          'ListPrice <= 0',    lambda s: s <= 0)
sold, _   = flag_invalid(sold, 'OriginalListPrice', 'SOLD',
                          'OriginalListPrice <= 0',    lambda s: s <= 0)
sold, _   = flag_invalid(sold, 'OriginalListPrice', 'SOLD',
                          'OriginalListPrice > $50M',  lambda s: s > 50_000_000)
sold, _   = flag_invalid(sold, 'LivingArea', 'SOLD',
                          'LivingArea <= 0',  lambda s: s <= 0)
sold, _   = flag_invalid(sold, 'DaysOnMarket', 'SOLD',
                          'DaysOnMarket < 0', lambda s: s < 0)

listed, _ = flag_invalid(listed, 'ListPrice', 'LISTED',
                          'ListPrice <= 0',   lambda s: s <= 0)
listed, _ = flag_invalid(listed, 'ListPrice', 'LISTED',
                          'ListPrice > $50M', lambda s: s > 50_000_000)
listed, _ = flag_invalid(listed, 'LivingArea', 'LISTED',
                          'LivingArea <= 0',  lambda s: s <= 0)

# Composite any-flag column (numeric + all date flags)
price_flags  = [c for c in sold.columns if c.endswith('_invalid_flag')]
date_flag_cols = [flag_col, flag_purchase, flag_neg]
all_sold_flags = price_flags + [c for c in date_flag_cols if c in sold.columns]
sold['any_invalid_flag'] = sold[all_sold_flags].max(axis=1).astype(int)
n_any = int(sold['any_invalid_flag'].sum())
print(f"\n  SOLD — total records with at least one quality flag: {n_any:,}")

listed_flags = [c for c in listed.columns if c.endswith('_invalid_flag')]
listed['any_invalid_flag'] = listed[listed_flags].max(axis=1).astype(int)
n_any_listed = int(listed['any_invalid_flag'].sum())
print(f"  LISTED — total records with at least one quality flag: {n_any_listed:,}")

# ============================================================
# 5. GEOGRAPHIC CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: GEOGRAPHIC / LAT-LON AUDIT")
print("=" * 70)

geo_stats = {}
for label, df in [("sold", sold), ("listed", listed)]:
    for col in ['Latitude', 'Longitude', 'latfilled', 'lonfilled',
                'PostalCode', 'City', 'CountyOrParish', 'StateOrProvince']:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            print(f"  [{label.upper()}] {col:25s}  {null_pct:5.1f}% null")
            geo_stats[f"{label}_{col}"] = round(null_pct, 2)

# Geographic quality flags
print("\n  --- Geographic quality flags ---")
for label, df in [("sold", sold), ("listed", listed)]:
    lat = pd.to_numeric(df.get('Latitude', pd.Series(dtype=float)), errors='coerce')
    lon = pd.to_numeric(df.get('Longitude', pd.Series(dtype=float)), errors='coerce')
    n_null_geo  = int((lat.isna() | lon.isna()).sum())
    n_zero_lat  = int((lat == 0).sum())
    n_zero_lon  = int((lon == 0).sum())
    n_pos_lon   = int((lon > 0).sum())    # CA should be negative longitude
    print(f"  [{label.upper()}] null lat or lon          : {n_null_geo:,}")
    print(f"  [{label.upper()}] lat == 0 (sentinel)      : {n_zero_lat:,}")
    print(f"  [{label.upper()}] lon == 0 (sentinel)      : {n_zero_lon:,}")
    print(f"  [{label.upper()}] lon > 0  (wrong side CA) : {n_pos_lon:,}")
    geo_stats[f"{label}_null_geo"]  = n_null_geo
    geo_stats[f"{label}_zero_lat"]  = n_zero_lat
    geo_stats[f"{label}_zero_lon"]  = n_zero_lon
    geo_stats[f"{label}_pos_lon"]   = n_pos_lon

    df['geo_null_flag']     = (lat.isna() | lon.isna()).astype(int)
    df['geo_sentinel_flag'] = ((lat == 0) | (lon == 0)).astype(int)
    df['geo_invalid_flag']  = (lon > 0).astype(int)

# ============================================================
# 6. CLEANING SUMMARY STATS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: SAVING CLEANED DATASETS & SUMMARY FILES")
print("=" * 70)

summary = {
    "sold": {
        "raw_rows": sold_raw_rows,
        "raw_cols": sold_raw_cols,
        "clean_rows": len(sold),
        "clean_cols": sold.shape[1],
        "cols_dropped": len(sold_dropped_cols),
        "dropped_col_names": sold_dropped_cols,
        "date_flags": int(n_flagged),
        "purchase_after_close_flags": n_purchase_flag,
        "negative_timeline_flags": n_neg_flag,
        "any_invalid_flag": n_any,
        "date_parse_results": sold_date_results,
    },
    "listed": {
        "raw_rows": listed_raw_rows,
        "raw_cols": listed_raw_cols,
        "clean_rows": len(listed),
        "clean_cols": listed.shape[1],
        "cols_dropped": len(listed_dropped_cols),
        "dropped_col_names": listed_dropped_cols,
        "any_invalid_flag": n_any_listed,
        "date_parse_results": listed_date_results,
    },
    "geo_stats": geo_stats,
}

with open(os.path.join(DATA_DIR, "cleaning_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved: cleaning_summary.json")

pd.DataFrame(cleaning_log).to_csv(
    os.path.join(DATA_DIR, "numeric_flag_log.csv"), index=False
)
print("  Saved: numeric_flag_log.csv")

# Save cleaned datasets (with flags, but all rows kept — hard removal in Wk7)
sold_path   = os.path.join(DATA_DIR, "sold_clean_flagged.csv")
listed_path = os.path.join(DATA_DIR, "listed_clean_flagged.csv")

# Convert datetime cols back to string for CSV portability
for col in sold.select_dtypes(include='datetime64').columns:
    sold[col] = sold[col].dt.strftime('%Y-%m-%d')
for col in listed.select_dtypes(include='datetime64').columns:
    listed[col] = listed[col].dt.strftime('%Y-%m-%d')

sold.to_csv(sold_path,     index=False, encoding='utf-8-sig')
listed.to_csv(listed_path, index=False, encoding='utf-8-sig')

print(f"  Saved: sold_clean_flagged.csv   ({len(sold):,} rows × {sold.shape[1]} cols)")
print(f"  Saved: listed_clean_flagged.csv ({len(listed):,} rows × {listed.shape[1]} cols)")

print("\n" + "#" * 70)
print("  WEEKS 4-5 CLEANING DELIVERABLE — COMPLETE")
print("#" * 70)
