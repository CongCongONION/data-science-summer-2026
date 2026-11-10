"""
listed_analysis.py
Week 1 - Monthly Dataset Aggregation (Listed Transactions)

This script automatically scans all CSV files in the Listing data folder,
concatenates them into a single dataset, filters for PropertyType == 'Residential',
and saves the result as a clean CSV.

Author: Ruoyu Wang
Date: April 2026
"""

import pandas as pd
import os
import glob

# ============================================================
# CONFIGURATION - Update these paths to match your local setup
# ============================================================

DATA_DIR = r"D:\IDXexchange\raw\Listing"
OUTPUT_FILE = r"D:\IDXexchange\week1\combined_listed.csv"

# ============================================================
# STEP 1: Scan all CSV files in the Listing folder
# ============================================================

print("=" * 60)
print("STEP 1: Scanning Listing CSV files")
print("=" * 60)

listing_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

if not listing_files:
    print(f"[ERROR] No CSV files found in {DATA_DIR}. Please check the path.")
    exit()

print(f"\nFound {len(listing_files)} files:")
for f in listing_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"  - {os.path.basename(f)}  ({size_mb:.1f} MB)")

# ============================================================
# STEP 2: Read and concatenate all files
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Reading and concatenating all files")
print("=" * 60)

dfs = []
for filepath in listing_files:
    df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')
    print(f"  {os.path.basename(filepath)}: {len(df):,} rows, {len(df.columns)} columns")
    dfs.append(df)

listings = pd.concat(dfs, ignore_index=True)

print(f"\n>>> Total rows after concatenation: {len(listings):,}")
print(f">>> Total columns: {len(listings.columns)}")

# ============================================================
# STEP 3: PropertyType distribution (before filtering)
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: PropertyType distribution (before filtering)")
print("=" * 60)

print(f"\nTotal rows before Residential filter: {len(listings):,}")
print("\nPropertyType breakdown:")
for pt, count in listings['PropertyType'].value_counts().items():
    pct = count / len(listings) * 100
    print(f"  {pt}: {count:,} ({pct:.1f}%)")

# ============================================================
# STEP 4: Filter for Residential properties only
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Filtering for PropertyType == 'Residential'")
print("=" * 60)

listings = listings[listings['PropertyType'] == 'Residential'].copy()

print(f">>> Total rows after Residential filter: {len(listings):,}")

# ============================================================
# STEP 5: Data overview
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Data overview")
print("=" * 60)

print(f"\nShape: {listings.shape}")
print(f"\nColumn list:\n{list(listings.columns)}")
print(f"\nData types:\n{listings.dtypes}")
print(f"\nFirst 5 rows:")
print(listings.head())

# ============================================================
# STEP 6: Save combined dataset
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Saving combined Residential listing dataset")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
listings.to_csv(OUTPUT_FILE, index=False)

print(f">>> Saved to: {OUTPUT_FILE}")
print(f">>> Final row count: {len(listings):,}")
print(f">>> Final column count: {len(listings.columns)}")

print("\n" + "=" * 60)
print("DONE - Listing dataset aggregation complete")
print("=" * 60)
