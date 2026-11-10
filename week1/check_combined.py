"""
check_combined.py
Post-Aggregation Validation Script

Validates the combined_sold.csv and combined_listed.csv files against
Week 1 deliverable requirements:

1. File existence and size
2. All rows are PropertyType == 'Residential'
3. Date range covers January 2024 through March 2026
4. No duplicate records (based on ListingKey)
5. Monthly row count distribution
6. Column completeness and null analysis
7. Key numeric field statistics (ClosePrice, ListPrice, LivingArea, DaysOnMarket)
8. PropertySubType breakdown
9. Top cities and counties by volume

Author: Ruoyu Wang
Date: April 2026
"""

import pandas as pd
import os

# ============================================================
# CONFIGURATION - Update these paths to match your local setup
# ============================================================

SOLD_FILE = r"D:\IDXexchange\week1\combined_sold.csv"
LISTED_FILE = r"D:\IDXexchange\week1\combined_listed.csv"

# ============================================================
# Validation function
# ============================================================

def validate_dataset(filepath, dataset_name, date_col):
    """Run all validation checks on a combined dataset."""

    print("\n" + "#" * 70)
    print(f"  VALIDATING: {dataset_name}")
    print(f"  File: {filepath}")
    print("#" * 70)

    # ----------------------------------------------------------
    # CHECK 1: File existence and size
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 1: File existence and size")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"  [FAIL] File not found: {filepath}")
        print("  Please run the aggregation script first.")
        return
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  [PASS] File exists ({size_mb:.1f} MB)")

    df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # ----------------------------------------------------------
    # CHECK 2: PropertyType == 'Residential' only
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 2: PropertyType filter verification")
    print("=" * 60)

    if 'PropertyType' in df.columns:
        unique_types = df['PropertyType'].unique()
        if len(unique_types) == 1 and unique_types[0] == 'Residential':
            print(f"  [PASS] All {len(df):,} rows are PropertyType == 'Residential'")
        else:
            print(f"  [FAIL] Found non-Residential records:")
            for pt in unique_types:
                count = (df['PropertyType'] == pt).sum()
                print(f"    {pt}: {count:,}")
    else:
        print("  [FAIL] PropertyType column not found")

    # ----------------------------------------------------------
    # CHECK 3: Date range coverage (Jan 2024 - Mar 2026)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CHECK 3: Date range coverage ({date_col})")
    print("=" * 60)

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = dates.notna().sum()
        null_dates = dates.isna().sum()

        print(f"  Valid dates: {valid_dates:,}")
        print(f"  Null/invalid dates: {null_dates:,}")
        print(f"  Earliest: {dates.min()}")
        print(f"  Latest: {dates.max()}")

        # Check expected range
        earliest = dates.min()
        latest = dates.max()

        if earliest.year == 2024 and earliest.month == 1:
            print(f"  [PASS] Data starts from January 2024")
        else:
            print(f"  [WARNING] Expected start: January 2024, got: {earliest}")

        if latest.year == 2026 and latest.month >= 2:
            print(f"  [PASS] Data includes 2026 records (latest: {latest.strftime('%Y-%m')})")
        else:
            print(f"  [WARNING] Expected data through at least Feb 2026, latest: {latest}")

        # Check for gaps
        monthly = dates.dt.to_period('M').value_counts().sort_index()
        expected_months = pd.period_range('2024-01', '2026-03', freq='M')
        missing_months = [m for m in expected_months if m not in monthly.index]

        if missing_months:
            print(f"  [WARNING] Missing months: {missing_months}")
        else:
            print(f"  [PASS] No missing months detected (Jan 2024 - Mar 2026)")
    else:
        print(f"  [FAIL] {date_col} column not found")

    # ----------------------------------------------------------
    # CHECK 4: Duplicate records
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 4: Duplicate record check")
    print("=" * 60)

    if 'ListingKey' in df.columns:
        total = len(df)
        unique = df['ListingKey'].nunique()
        duplicates = total - unique
        if duplicates == 0:
            print(f"  [PASS] No duplicate ListingKey values ({unique:,} unique records)")
        else:
            print(f"  [WARNING] {duplicates:,} duplicate ListingKey values found")
            print(f"    Total rows: {total:,}")
            print(f"    Unique ListingKeys: {unique:,}")
    else:
        print("  [INFO] ListingKey column not found, skipping duplicate check")

    # ----------------------------------------------------------
    # CHECK 5: Monthly row count distribution
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 5: Monthly row count distribution")
    print("=" * 60)

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        monthly = dates.dt.to_period('M').value_counts().sort_index()

        print(f"\n  {'Month':<12} {'Count':>10} {'Bar'}")
        print("  " + "-" * 50)

        max_count = monthly.max()
        for period, count in monthly.items():
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            print(f"  {str(period):<12} {count:>10,} {bar}")

        print(f"\n  Total: {monthly.sum():,}")
        print(f"  Monthly average: {monthly.mean():,.0f}")
        print(f"  Monthly min: {monthly.min():,} ({monthly.idxmin()})")
        print(f"  Monthly max: {monthly.max():,} ({monthly.idxmax()})")

    # ----------------------------------------------------------
    # CHECK 6: Column completeness and null analysis
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 6: Column null analysis")
    print("=" * 60)

    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(1)
    null_summary = pd.DataFrame({
        'null_count': null_counts,
        'null_pct': null_pct
    }).sort_values('null_pct', ascending=False)

    # Columns with > 90% null
    high_null = null_summary[null_summary['null_pct'] > 90]
    if len(high_null) > 0:
        print(f"\n  Columns with > 90% null ({len(high_null)} columns):")
        for col, row in high_null.iterrows():
            print(f"    {col}: {row['null_count']:,.0f} nulls ({row['null_pct']}%)")
    else:
        print(f"\n  [PASS] No columns with > 90% null")

    # Columns with 0% null
    zero_null = null_summary[null_summary['null_pct'] == 0]
    print(f"\n  Columns with 0% null: {len(zero_null)} / {len(df.columns)}")

    # Critical fields null check
    print(f"\n  Critical field null rates:")
    critical_fields = ['PropertyType', 'City', 'PostalCode', 'CountyOrParish',
                       'UnparsedAddress', 'ListPrice', 'LivingArea',
                       'BedroomsTotal', 'BathroomsTotalInteger', 'YearBuilt']
    for col in critical_fields:
        if col in df.columns:
            nc = df[col].isnull().sum()
            np_ = nc / len(df) * 100
            status = "[PASS]" if np_ < 5 else "[WARNING]"
            print(f"    {status} {col}: {nc:,} nulls ({np_:.1f}%)")

    # ----------------------------------------------------------
    # CHECK 7: Key numeric field statistics
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 7: Key numeric field statistics")
    print("=" * 60)

    numeric_fields = {
        'ClosePrice': 'Close price of the property',
        'ListPrice': 'Current list price',
        'OriginalListPrice': 'Original list price',
        'LivingArea': 'Total livable area (sq ft)',
        'DaysOnMarket': 'Days on market',
        'BedroomsTotal': 'Total bedrooms',
        'BathroomsTotalInteger': 'Total bathrooms',
        'YearBuilt': 'Year built',
        'LotSizeSquareFeet': 'Lot size (sq ft)',
        'TaxAnnualAmount': 'Annual tax amount'
    }

    for field, desc in numeric_fields.items():
        if field not in df.columns:
            continue
        s = pd.to_numeric(df[field], errors='coerce')
        non_null = s.notna().sum()
        if non_null == 0:
            continue

        print(f"\n  {field} ({desc}):")
        print(f"    Count:  {non_null:,}")
        print(f"    Min:    {s.min():,.2f}")
        print(f"    25th:   {s.quantile(0.25):,.2f}")
        print(f"    Median: {s.median():,.2f}")
        print(f"    75th:   {s.quantile(0.75):,.2f}")
        print(f"    Max:    {s.max():,.2f}")
        print(f"    Mean:   {s.mean():,.2f}")
        print(f"    Std:    {s.std():,.2f}")

        # Flag potential outliers
        if field in ['ClosePrice', 'ListPrice', 'OriginalListPrice']:
            if s.min() <= 0:
                neg_count = (s <= 0).sum()
                print(f"    [WARNING] {neg_count:,} records with {field} <= 0")
            if s.max() > 50_000_000:
                extreme = (s > 50_000_000).sum()
                print(f"    [WARNING] {extreme:,} records with {field} > $50M")

        if field == 'DaysOnMarket' and s.min() < 0:
            neg_count = (s < 0).sum()
            print(f"    [WARNING] {neg_count:,} records with negative DaysOnMarket")

        if field == 'LivingArea' and s.min() <= 0:
            zero_count = (s <= 0).sum()
            print(f"    [WARNING] {zero_count:,} records with LivingArea <= 0")

    # ----------------------------------------------------------
    # CHECK 8: PropertySubType breakdown
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 8: PropertySubType breakdown")
    print("=" * 60)

    if 'PropertySubType' in df.columns:
        subtype_counts = df['PropertySubType'].value_counts()
        print(f"\n  {'PropertySubType':<35} {'Count':>10} {'Pct':>8}")
        print("  " + "-" * 55)
        for st, count in subtype_counts.items():
            pct = count / len(df) * 100
            print(f"  {str(st):<35} {count:>10,} {pct:>7.1f}%")

    # ----------------------------------------------------------
    # CHECK 9: Top cities and counties
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 9: Top 10 cities and counties by volume")
    print("=" * 60)

    if 'City' in df.columns:
        top_cities = df['City'].value_counts().head(10)
        print(f"\n  Top 10 Cities:")
        for city, count in top_cities.items():
            pct = count / len(df) * 100
            print(f"    {city}: {count:,} ({pct:.1f}%)")

    if 'CountyOrParish' in df.columns:
        top_counties = df['CountyOrParish'].value_counts().head(10)
        print(f"\n  Top 10 Counties:")
        for county, count in top_counties.items():
            pct = count / len(df) * 100
            print(f"    {county}: {count:,} ({pct:.1f}%)")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY - {dataset_name}")
    print("=" * 60)
    print(f"  File: {filepath}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        print(f"  Date range: {dates.min()} ~ {dates.max()}")
    print(f"  Columns with > 90% null: {len(high_null)}")
    print(f"  Columns with 0% null: {len(zero_null)}")


# ============================================================
# Run validation on both files
# ============================================================

if __name__ == "__main__":
    validate_dataset(SOLD_FILE, "Combined Sold Dataset", "CloseDate")
    validate_dataset(LISTED_FILE, "Combined Listed Dataset", "ListingContractDate")

    print("\n" + "#" * 70)
    print("  ALL VALIDATIONS COMPLETE")
    print("#" * 70)
