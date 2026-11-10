"""
check_data.py
Data Validation Script - Automatically scans all CSV files in the target folder

Checks:
1. File existence and size
2. Row and column counts per file
3. Column name consistency across files
4. Key field data quality (PropertyType, dates, prices)
5. Summary statistics

Author: Ruoyu Wang
Date: April 2026
"""

import pandas as pd
import os
import glob

# ============================================================
# CONFIGURATION - Update this path to match your local setup
# ============================================================

DATA_DIR = r"D:\IDXexchange\raw"

# ============================================================
# STEP 1: Scan all CSV files in the folder
# ============================================================

print("=" * 70)
print(f"Scanning directory: {DATA_DIR}")
print("=" * 70)

all_csv = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))

if not all_csv:
    print("[ERROR] No CSV files found. Please check the path.")
    exit()

sold_files = [f for f in all_csv if "Sold" in os.path.basename(f)]
listing_files = [f for f in all_csv if "Listing" in os.path.basename(f)]
other_files = [f for f in all_csv if f not in sold_files and f not in listing_files]

print(f"\nTotal CSV files found: {len(all_csv)}")
print(f"  Sold files: {len(sold_files)}")
print(f"  Listing files: {len(listing_files)}")
if other_files:
    print(f"  Other files: {len(other_files)}")
    for f in other_files:
        print(f"    - {os.path.basename(f)}")

# ============================================================
# STEP 2: File overview (rows, columns, size)
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: File overview")
print("=" * 70)

print(f"\n{'File':<35} {'Size(MB)':>8} {'Rows':>10} {'Cols':>6}")
print("-" * 65)

for label, files in [("--- Sold ---", sold_files), ("--- Listing ---", listing_files)]:
    print(f"\n{label}")
    total_rows = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        row_count = sum(1 for _ in open(filepath, encoding='latin-1')) - 1
        df_header = pd.read_csv(filepath, low_memory=False, nrows=0, encoding='latin-1')
        num_cols = len(df_header.columns)
        total_rows += row_count
        print(f"  {filename:<33} {size_mb:>8.1f} {row_count:>10,} {num_cols:>6}")
    print(f"  {'TOTAL':<33} {'':>8} {total_rows:>10,}")

# ============================================================
# STEP 3: Column name consistency check
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Column name consistency check")
print("=" * 70)

for label, files in [("Sold", sold_files), ("Listing", listing_files)]:
    if not files:
        continue

    all_columns = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, low_memory=False, nrows=0, encoding='latin-1')
        cols = tuple(sorted(df.columns.tolist()))
        if cols not in all_columns:
            all_columns[cols] = []
        all_columns[cols].append(filename)

    if len(all_columns) == 1:
        col_count = len(list(all_columns.keys())[0])
        print(f"\n  [OK] {label}: All {len(files)} files have consistent columns ({col_count} columns)")
    else:
        print(f"\n  [WARNING] {label}: Found {len(all_columns)} different column structures")
        col_sets = list(all_columns.keys())
        for i, (cols, filenames) in enumerate(all_columns.items()):
            print(f"\n     Structure {i+1} ({len(cols)} columns), {len(filenames)} files:")
            for f in filenames:
                print(f"       - {f}")
        if len(col_sets) > 1:
            base = set(col_sets[0])
            for i in range(1, len(col_sets)):
                other = set(col_sets[i])
                extra = other - base
                missing = base - other
                if extra:
                    print(f"\n     Structure {i+1} extra columns vs Structure 1: {extra}")
                if missing:
                    print(f"     Structure {i+1} missing columns vs Structure 1: {missing}")

# ============================================================
# STEP 4: Key field data quality check
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Key field data quality check")
print("=" * 70)

for label, files, date_col in [
    ("Sold", sold_files, "CloseDate"),
    ("Listing", listing_files, "ListingContractDate")
]:
    if not files:
        continue

    print(f"\n{'='*40}")
    print(f"  {label} files")
    print(f"{'='*40}")

    for filepath in files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')

        print(f"\n  --- {filename} ({len(df):,} rows) ---")

        # PropertyType distribution
        if 'PropertyType' in df.columns:
            print(f"  PropertyType:")
            for pt, count in df['PropertyType'].value_counts().items():
                pct = count / len(df) * 100
                print(f"    {pt}: {count:,} ({pct:.1f}%)")

        # Date range
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            valid = dates.notna().sum()
            print(f"  {date_col}: {dates.min()} ~ {dates.max()} ({valid:,} valid)")

        # Key numeric field statistics
        numeric_fields = ['ClosePrice', 'ListPrice', 'OriginalListPrice',
                          'LivingArea', 'DaysOnMarket']
        for field in numeric_fields:
            if field in df.columns:
                s = pd.to_numeric(df[field], errors='coerce')
                non_null = s.notna().sum()
                if non_null > 0:
                    print(f"  {field}: min={s.min():,.0f}  median={s.median():,.0f}  "
                          f"max={s.max():,.0f}  nulls={len(df)-non_null}")

        # Critical field null check
        critical = ['PropertyType', 'City', 'PostalCode', 'CountyOrParish',
                     'UnparsedAddress']
        null_issues = []
        for col in critical:
            if col in df.columns:
                null_pct = df[col].isnull().sum() / len(df) * 100
                if null_pct > 5:
                    null_issues.append(f"{col}({null_pct:.0f}%)")
        if null_issues:
            print(f"  [WARNING] High null fields: {', '.join(null_issues)}")
        else:
            print(f"  [OK] All critical fields have < 5% nulls")

# ============================================================
# STEP 5: Summary
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Summary")
print("=" * 70)

for label, files, date_col in [
    ("Sold", sold_files, "CloseDate"),
    ("Listing", listing_files, "ListingContractDate")
]:
    if not files:
        continue

    all_dfs = [pd.read_csv(f, low_memory=False, encoding='latin-1') for f in files]
    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  {label} summary:")
    print(f"    Files: {len(files)}")
    print(f"    Total rows: {len(combined):,}")
    print(f"    Total columns: {len(combined.columns)}")

    if date_col in combined.columns:
        dates = pd.to_datetime(combined[date_col], errors='coerce')
        monthly = dates.dt.to_period('M').value_counts().sort_index()
        print(f"    Date range: {dates.min()} ~ {dates.max()}")
        print(f"    Monthly row counts:")
        for period, count in monthly.items():
            print(f"      {period}: {count:,}")

print("\n" + "=" * 70)
print("Data check complete!")
print("=" * 70)
