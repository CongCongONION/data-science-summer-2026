"""
outlier_detection.py
Week 7 Deliverable: Outlier Detection and Data Quality

Method: Interquartile Range (IQR) — tiered approach per handbook:
  1. Apply business rules (ClosePrice<=0, LivingArea<=0 are always invalid)
  2. Apply IQR outlier flags to: ClosePrice, LivingArea, DaysOnMarket,
     ppsf, listing_to_contract_days, contract_to_close_days
  3. Save full flagged dataset (all rows preserved)
  4. Save clean filtered dataset (rows with no outlier flags)
  5. Before/after comparison: row counts and median values

Author: Ruoyu Wang
Date:   May 2026
"""

import os
import json
import pandas as pd
import numpy as np

WEEK6_DATA = r"D:\IDXexchange\week6\data"
WEEK7_DIR  = r"D:\IDXexchange\week7"
DATA_DIR   = os.path.join(WEEK7_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. LOAD
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING WEEK-6 FEATURE-ENGINEERED DATASET")
print("=" * 70)

sold = pd.read_csv(
    os.path.join(WEEK6_DATA, "sold_with_features.csv"),
    low_memory=False,
)
print(f"  Loaded: {len(sold):,} rows x {sold.shape[1]} columns")

NUMERIC = [
    "ClosePrice", "LivingArea", "DaysOnMarket", "ppsf",
    "listing_to_contract_days", "contract_to_close_days",
    "OriginalListPrice", "ListPrice",
]
for col in NUMERIC:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors="coerce")

# ============================================================
# 2. BUSINESS RULES (always-invalid records)
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: BUSINESS RULES — ALWAYS-INVALID RECORDS")
print("=" * 70)

business_flags = {}

def biz_flag(df, flag_col, mask, label):
    df[flag_col] = mask.astype(int)
    n = int(mask.sum())
    print(f"  {flag_col:45s}  {n:,} records")
    business_flags[flag_col] = n
    return df

sold = biz_flag(sold, "biz_close_price_invalid",
    sold["ClosePrice"].isna() | (sold["ClosePrice"] <= 0),
    "ClosePrice <= 0 or null")

sold = biz_flag(sold, "biz_living_area_invalid",
    sold["LivingArea"].isna() | (sold["LivingArea"] <= 0),
    "LivingArea <= 0 or null")

sold = biz_flag(sold, "biz_dom_invalid",
    sold["DaysOnMarket"] < 0,
    "DaysOnMarket < 0")

sold["any_biz_invalid"] = (
    sold[["biz_close_price_invalid", "biz_living_area_invalid", "biz_dom_invalid"]]
    .max(axis=1)
)
n_biz = int(sold["any_biz_invalid"].sum())
print(f"\n  Total business-rule violations: {n_biz:,}")

# Work on business-valid records for IQR calculation
valid_base = sold[sold["any_biz_invalid"] == 0].copy()
print(f"  Records available for IQR analysis: {len(valid_base):,}")

# ============================================================
# 3. IQR OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: IQR OUTLIER DETECTION")
print("=" * 70)

IQR_FIELDS = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "ppsf",
    "listing_to_contract_days",
    "contract_to_close_days",
]

iqr_stats  = {}
flag_cols  = []

for col in IQR_FIELDS:
    if col not in sold.columns:
        print(f"  {col}: NOT FOUND — skip")
        continue

    series = valid_base[col].dropna()
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR

    flag = f"{col}_outlier_flag"
    sold[flag] = 0
    mask = (
        sold[col].notna() &
        (sold["any_biz_invalid"] == 0) &
        ((sold[col] < lo) | (sold[col] > hi))
    )
    sold.loc[mask, flag] = 1
    n_out = int(mask.sum())
    flag_cols.append(flag)

    pct_out = n_out / len(valid_base) * 100
    print(f"  {col:35s}  Q1={Q1:,.2f}  Q3={Q3:,.2f}  IQR={IQR:,.2f}")
    print(f"    Bounds: [{lo:,.2f}, {hi:,.2f}]  Flagged: {n_out:,} ({pct_out:.2f}%)")

    iqr_stats[col] = {
        "Q1": round(float(Q1), 2), "Q3": round(float(Q3), 2),
        "IQR": round(float(IQR), 2),
        "lower_bound": round(float(lo), 2),
        "upper_bound": round(float(hi), 2),
        "n_outliers": n_out,
        "pct_outliers": round(pct_out, 4),
        "median_before": round(float(series.median()), 2),
    }

# Composite outlier flag (any field flagged)
sold["any_outlier_flag"] = sold[flag_cols].max(axis=1).astype(int)
n_any = int(sold["any_outlier_flag"].sum())
print(f"\n  Records with ANY IQR outlier flag: {n_any:,} ({n_any/len(sold)*100:.2f}%)")

# Master exclusion flag = business invalid OR any outlier
sold["exclude_flag"] = ((sold["any_biz_invalid"] == 1) | (sold["any_outlier_flag"] == 1)).astype(int)
n_excl = int(sold["exclude_flag"].sum())
print(f"  Records excluded (biz invalid + outliers): {n_excl:,} ({n_excl/len(sold)*100:.2f}%)")

# ============================================================
# 4. BEFORE / AFTER COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: BEFORE / AFTER COMPARISON")
print("=" * 70)

clean = sold[sold["exclude_flag"] == 0].copy()
n_before = len(sold)
n_after  = len(clean)
removed  = n_before - n_after

comparison = {}
compare_fields = ["ClosePrice", "LivingArea", "DaysOnMarket", "ppsf"]

print(f"\n  Rows before: {n_before:,}")
print(f"  Rows after : {n_after:,}  (removed {removed:,}, {removed/n_before*100:.2f}%)")
print()
print(f"  {'Field':<35s}  {'Median Before':>14}  {'Median After':>13}  {'Delta':>10}")
print("  " + "-" * 78)
for col in compare_fields:
    if col not in sold.columns:
        continue
    med_b = sold[col].median()
    med_a = clean[col].median()
    delta = med_a - med_b
    sign  = "+" if delta >= 0 else ""
    print(f"  {col:<35s}  {med_b:>14,.2f}  {med_a:>13,.2f}  {sign}{delta:>9,.2f}")
    comparison[col] = {
        "median_before": round(float(med_b), 2) if not pd.isna(med_b) else None,
        "median_after":  round(float(med_a), 2) if not pd.isna(med_a) else None,
        "delta": round(float(delta), 2) if not pd.isna(delta) else None,
    }

# ============================================================
# 5. PERCENTILE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: PERCENTILE ANALYSIS (CLOSE PRICE)")
print("=" * 70)

pcts = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
cp_series = sold["ClosePrice"].dropna()
print(f"  {'Percentile':>12}   {'ClosePrice':>15}")
percentiles = {}
for p in pcts:
    v = float(cp_series.quantile(p))
    print(f"  {p*100:>11.0f}%   ${v:>14,.0f}")
    percentiles[f"p{int(p*100)}"] = round(v, 2)

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: SAVING OUTPUTS")
print("=" * 70)

# Full flagged dataset (all rows preserved, flags added)
flagged_path = os.path.join(DATA_DIR, "sold_outlier_flagged.csv")
sold.to_csv(flagged_path, index=False, encoding="utf-8-sig")
print(f"  Saved: sold_outlier_flagged.csv  ({len(sold):,} rows x {sold.shape[1]} cols)")

# Clean analysis-ready dataset
clean_path = os.path.join(DATA_DIR, "sold_analysis_ready.csv")
clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
print(f"  Saved: sold_analysis_ready.csv   ({len(clean):,} rows x {clean.shape[1]} cols)")

# Summary JSON
summary = {
    "source_file":     "sold_with_features.csv",
    "rows_before":     n_before,
    "rows_after":      n_after,
    "rows_removed":    removed,
    "pct_removed":     round(removed / n_before * 100, 4),
    "business_flags":  business_flags,
    "n_biz_invalid":   n_biz,
    "n_iqr_outliers":  n_any,
    "n_excluded_total": n_excl,
    "iqr_stats":       iqr_stats,
    "comparison":      comparison,
    "close_price_percentiles": percentiles,
}
with open(os.path.join(DATA_DIR, "outlier_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("  Saved: outlier_summary.json")

print("\n" + "#" * 70)
print("  WEEK 7 OUTLIER DETECTION DELIVERABLE -- COMPLETE")
print("#" * 70)
