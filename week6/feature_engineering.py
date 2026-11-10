"""
feature_engineering.py
Week 6 Deliverable: Feature Engineering and Market Metrics

Covers all handbook-specified engineered features:
  1. price_ratio              ClosePrice / OriginalListPrice
  2. close_to_orig_list_ratio ClosePrice / OriginalListPrice  (alias, same metric)
  3. ppsf                     ClosePrice / LivingArea
  4. year / month / yr_mo     Derived from CloseDate
  5. listing_to_contract_days PurchaseContractDate - ListingContractDate
  6. contract_to_close_days   CloseDate - PurchaseContractDate
  7. days_on_market           Raw DaysOnMarket field (no change)

Segment analysis:
  - By CountyOrParish
  - By PropertySubType
  - By ListOfficeName  (top 20)

Author: Ruoyu Wang
Date:   May 2026
"""

import os
import json
import pandas as pd
import numpy as np

WEEK5_DATA = r"D:\IDXexchange\week5\data"
WEEK6_DIR  = r"D:\IDXexchange\week6"
DATA_DIR   = os.path.join(WEEK6_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. LOAD CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING WEEK-5 FINAL CLEAN DATASET (SOLD)")
print("=" * 70)

sold = pd.read_csv(
    os.path.join(WEEK5_DATA, "sold_final_clean.csv"),
    low_memory=False,
)
print(f"  Loaded: {len(sold):,} rows × {sold.shape[1]} columns")

# Cast numeric fields
NUMERIC = [
    "ClosePrice", "OriginalListPrice", "ListPrice",
    "LivingArea", "LotSizeAcres", "DaysOnMarket",
    "BedroomsTotal", "BathroomsTotalInteger", "YearBuilt",
]
for col in NUMERIC:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors="coerce")

# Parse date fields
DATE_COLS = ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
for col in DATE_COLS:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")

print(f"  Date fields parsed: {DATE_COLS}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

feature_log = []

def add_feature(df, name, series, description, formula):
    df[name] = series
    n_valid = int(df[name].notna().sum())
    pct     = n_valid / len(df) * 100
    print(f"  {name:35s}  valid={n_valid:,} ({pct:.1f}%)")
    feature_log.append({
        "feature": name, "description": description,
        "formula": formula, "valid_count": n_valid,
        "valid_pct": round(pct, 2),
    })

# 2a. Price Ratio (negotiation strength)
add_feature(
    sold, "price_ratio",
    sold["ClosePrice"] / sold["OriginalListPrice"].replace(0, np.nan),
    "ClosePrice / OriginalListPrice — measures negotiation strength",
    "ClosePrice / OriginalListPrice",
)

# 2b. Close-to-Original-List Ratio (alias with same formula, separate column for clarity)
add_feature(
    sold, "close_to_orig_list_ratio",
    sold["ClosePrice"] / sold["OriginalListPrice"].replace(0, np.nan),
    "Captures full price-reduction history from original ask",
    "ClosePrice / OriginalListPrice",
)

# 2c. Price Per Square Foot
add_feature(
    sold, "ppsf",
    sold["ClosePrice"] / sold["LivingArea"].replace(0, np.nan),
    "Price per square foot — normalises price across property sizes",
    "ClosePrice / LivingArea",
)

# 2d. Time-series keys from CloseDate
add_feature(sold, "close_year",  sold["CloseDate"].dt.year,  "Year of close",         "CloseDate.dt.year")
add_feature(sold, "close_month", sold["CloseDate"].dt.month, "Month of close (1-12)", "CloseDate.dt.month")

yr_mo = sold["CloseDate"].dt.to_period("M").astype(str).where(sold["CloseDate"].notna(), np.nan)
add_feature(sold, "yr_mo", yr_mo, "Year-month string (YYYY-MM) for time-series joins", "CloseDate.dt.to_period('M')")

# 2e. Listing-to-Contract Days
ltd = (sold["PurchaseContractDate"] - sold["ListingContractDate"]).dt.days
add_feature(
    sold, "listing_to_contract_days",
    ltd,
    "Days from listing to accepted offer (time-on-market until contract)",
    "PurchaseContractDate - ListingContractDate",
)

# 2f. Contract-to-Close Days (escrow period)
ctc = (sold["CloseDate"] - sold["PurchaseContractDate"]).dt.days
add_feature(
    sold, "contract_to_close_days",
    ctc,
    "Days from accepted offer to close of escrow",
    "CloseDate - PurchaseContractDate",
)

print(f"\n  Total new features added: {len(feature_log)}")

# ============================================================
# 3. VALIDATION CHECKS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: FEATURE VALIDATION")
print("=" * 70)

# Flag implausible engineered values
sold["price_ratio_flag"]   = ((sold["price_ratio"] < 0.3) | (sold["price_ratio"] > 2.0)).astype(int)
sold["ppsf_flag"]          = ((sold["ppsf"] < 10) | (sold["ppsf"] > 5000)).astype(int)
sold["ltc_days_flag"]      = ((sold["listing_to_contract_days"] < 0) | (sold["listing_to_contract_days"] > 1825)).astype(int)
sold["ctc_days_flag"]      = ((sold["contract_to_close_days"] < 0)   | (sold["contract_to_close_days"]   > 365)).astype(int)

for flag in ["price_ratio_flag", "ppsf_flag", "ltc_days_flag", "ctc_days_flag"]:
    n   = int(sold[flag].sum())
    pct = n / len(sold) * 100
    print(f"  {flag:35s}  {n:,} ({pct:.3f}%)")

# ============================================================
# 4. SEGMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: SEGMENT ANALYSIS")
print("=" * 70)

# Filter to records with valid key metrics
valid = sold[
    sold["price_ratio"].notna() &
    sold["ppsf"].notna() &
    sold["close_year"].notna() &
    (sold["ClosePrice"] > 0) &
    (sold["ppsf_flag"] == 0)
].copy()

print(f"  Valid records for segment analysis: {len(valid):,}")

AGG = {
    "ClosePrice":               ["count", "median", "mean"],
    "ppsf":                     ["median", "mean"],
    "price_ratio":              ["median", "mean"],
    "DaysOnMarket":             ["median", "mean"],
    "listing_to_contract_days": "median",
    "contract_to_close_days":   "median",
}

def safe_segment(df, col, agg_dict, label, top_n=None):
    df = df.copy()
    df[col] = df[col].fillna("Unknown")
    grp = df.groupby(col).agg(agg_dict).round(2)
    grp.columns = ["_".join(c).strip("_") for c in grp.columns]
    grp = grp.reset_index()
    if "ClosePrice_count" in grp.columns:
        grp = grp[grp["ClosePrice_count"] >= 10]
        grp = grp.sort_values("ClosePrice_count", ascending=False)
    if top_n:
        grp = grp.head(top_n)
    print(f"  {label}: {len(grp)} groups after filtering (min 10 records)")
    return grp

seg_county  = safe_segment(valid, "CountyOrParish", AGG, "By County")
seg_subtype = safe_segment(valid, "PropertySubType", AGG, "By PropertySubType")
seg_office  = safe_segment(valid, "ListOfficeName",  AGG, "By Listing Office", top_n=20)

# ── Monthly market trend ──────────────────────────────────────────────────

monthly_agg = {
    "ClosePrice": ["count", "median"],
    "ppsf":       "median",
    "price_ratio": "median",
    "DaysOnMarket": "median",
}
monthly = (
    valid[valid["yr_mo"].notna()]
    .groupby("yr_mo")
    .agg(monthly_agg)
    .round(2)
)
monthly.columns = ["_".join(c).strip("_") for c in monthly.columns]
monthly = monthly.reset_index().sort_values("yr_mo")

print(f"  Monthly trend: {len(monthly)} months")

# ============================================================
# 5. SAMPLE OUTPUT TABLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: SAMPLE OUTPUT (10 ROWS)")
print("=" * 70)

SHOW_COLS = [
    "ListingId", "CloseDate", "CountyOrParish", "PropertySubType",
    "ClosePrice", "OriginalListPrice", "LivingArea",
    "price_ratio", "ppsf", "DaysOnMarket",
    "listing_to_contract_days", "contract_to_close_days",
    "yr_mo",
]
show_cols = [c for c in SHOW_COLS if c in sold.columns]
sample = sold[show_cols].dropna(subset=["price_ratio", "ppsf"]).head(10)
print(sample.to_string(index=False))

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: SAVING OUTPUTS")
print("=" * 70)

# Main dataset with all features
sold_out = os.path.join(DATA_DIR, "sold_with_features.csv")
sold.to_csv(sold_out, index=False, encoding="utf-8-sig")
print(f"  Saved: sold_with_features.csv ({len(sold):,} rows × {sold.shape[1]} cols)")

# Segment tables
seg_county.to_csv( os.path.join(DATA_DIR, "segment_by_county.csv"),    index=False, encoding="utf-8-sig")
seg_subtype.to_csv(os.path.join(DATA_DIR, "segment_by_subtype.csv"),   index=False, encoding="utf-8-sig")
seg_office.to_csv( os.path.join(DATA_DIR, "segment_by_office.csv"),    index=False, encoding="utf-8-sig")
monthly.to_csv(    os.path.join(DATA_DIR, "monthly_market_trend.csv"), index=False, encoding="utf-8-sig")
print("  Saved: segment_by_county.csv, segment_by_subtype.csv, segment_by_office.csv")
print("  Saved: monthly_market_trend.csv")

# Feature metadata log
pd.DataFrame(feature_log).to_csv(
    os.path.join(DATA_DIR, "feature_log.csv"), index=False, encoding="utf-8-sig"
)
print("  Saved: feature_log.csv")

# Summary JSON
fe_summary = {
    "source_file": "sold_final_clean.csv",
    "total_rows": len(sold),
    "total_cols": sold.shape[1],
    "features_added": feature_log,
    "segment_county_groups":  len(seg_county),
    "segment_subtype_groups": len(seg_subtype),
    "segment_office_top20":   len(seg_office),
    "monthly_periods":        len(monthly),
}
with open(os.path.join(DATA_DIR, "feature_summary.json"), "w") as f:
    json.dump(fe_summary, f, indent=2, default=str)
print("  Saved: feature_summary.json")

# Expose for report generator
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  WEEK 6 FEATURE ENGINEERING DELIVERABLE — COMPLETE")
    print("#" * 70)
