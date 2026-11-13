"""
idx_pipeline.py — IDX Exchange / CRMLS analytics pipeline (independent reproduction)

Author : Anna
Date   : November 2026

What this is
------------
A from-scratch re-implementation of the IDX Exchange data-analyst project.
The *method* follows the team project specification — same data sources,
same definitions, same thresholds — but every line here was written for this
reproduction, and every number this code prints comes from the run that
executes it (nothing is carried over from anyone else's output).

Pipeline stages
---------------
  1  aggregate   Monthly CRMLS CSVs (Sold + Listing) -> one combined file each,
                 keeping PropertyType == "Residential" only.
  2  eda         Missing-value profile, numeric distributions, and mortgage
                 enrichment: FRED MORTGAGE30US weekly rates -> monthly means,
                 merged onto Sold (by CloseDate month) and Listing
                 (by ListingContractDate month).
  3  cleaning    Phase 1 cleaning: drop columns that are >90% empty, parse
                 dates, flag timeline contradictions and invalid numeric /
                 geographic values.
  4  geoclean    Phase 2 cleaning (reconstructed from the project spec):
                 California bounding-box check, dtype audit, final flag roll-up.
  5  features    Analytical features (price ratio, price per sq ft, timeline
                 durations, calendar parts) + segment tables + monthly trend.
  6  outliers    Business-rule invalids first, then 1.5x IQR outlier flags on
                 six fields; writes the analysis-ready Sold dataset.

Data sources
------------
  * CRMLS listing / sold records via the CoreLogic Trestle API, exported as
    monthly CSV files, Jan 2024 - Mar 2026. Raw exports are NOT in git;
    put them under data/raw/Sold and data/raw/Listing before running.
  * FRED series MORTGAGE30US (public, downloaded at run time).

Layout (all paths relative to this file — no absolute paths anywhere)
---------------------------------------------------------------------
  data/raw/Sold/*.csv, data/raw/Listing/*.csv   raw monthly exports
  data/work/                                    intermediate datasets
  output/                                       summaries, segments, final data

Usage
-----
  python idx_pipeline.py all          # stages 1-6 in order
  python idx_pipeline.py 1 3          # or just the stages you name
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

AUTHOR = "Anna"

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
WORK_DIR = os.path.join(ROOT, "data", "work")
OUT_DIR = os.path.join(ROOT, "output")

SUMMARY_PATH = os.path.join(OUT_DIR, "run_facts.json")

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _ensure_dirs() -> None:
    for d in (WORK_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


def _read(path: str) -> pd.DataFrame:
    # utf-8-sig strips any BOM so column names stay clean across stages;
    # latin-1 is the fallback for vendor exports with legacy characters
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def _save(df: pd.DataFrame, path: str) -> str:
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"    wrote {os.path.relpath(path, ROOT)}  ({len(df):,} x {df.shape[1]})")
    return path


def _record(stage: str, facts: dict) -> None:
    """Accumulate this run's facts so the report reflects *this* execution."""
    _ensure_dirs()
    book = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as fh:
            book = json.load(fh)
    book[stage] = facts
    book["_author"] = AUTHOR
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(book, fh, indent=2, ensure_ascii=False)


def _banner(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


# --------------------------------------------------------------------------
# stage 1 — aggregate monthly exports
# --------------------------------------------------------------------------


def _combine_monthly(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {folder}. Export the monthly CRMLS files there first."
        )
    print(f"  {len(files)} monthly files in {os.path.relpath(folder, ROOT)}")
    frames = []
    for fp in files:
        part = _read(fp)
        print(f"    {os.path.basename(fp):28s} {len(part):>7,} rows")
        frames.append(part)
    merged = pd.concat(frames, ignore_index=True)
    homes = merged.loc[merged["PropertyType"] == "Residential"].copy()
    print(f"  concatenated {len(merged):,} -> Residential only {len(homes):,}")
    return homes.reset_index(drop=True)


def stage_aggregate() -> None:
    _banner("Stage 1/6  aggregate monthly CRMLS exports")
    sold = _combine_monthly(os.path.join(RAW_DIR, "Sold"))
    listed = _combine_monthly(os.path.join(RAW_DIR, "Listing"))
    _save(sold, os.path.join(WORK_DIR, "combined_sold.csv"))
    _save(listed, os.path.join(WORK_DIR, "combined_listed.csv"))
    # provenance: the sample generator leaves a marker next to its files, so
    # every downstream result can state honestly what data produced it
    marker = os.path.join(RAW_DIR, "SAMPLE_DATA.txt")
    kind = "sample (synthetic, CRMLS schema)" if os.path.exists(marker) \
        else "real CRMLS exports"
    print(f"  data kind: {kind}")
    _record("aggregate", {"sold_rows": len(sold), "listed_rows": len(listed),
                          "data_kind": kind})


# --------------------------------------------------------------------------
# stage 2 — EDA profile + mortgage-rate enrichment
# --------------------------------------------------------------------------

PROFILE_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    tab = pd.DataFrame(
        {"null_count": df.isnull().sum(),
         "null_pct": (df.isnull().mean() * 100).round(2)}
    ).sort_values("null_pct", ascending=False)
    return tab


def _distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in PROFILE_FIELDS:
        if name not in df.columns:
            continue
        s = pd.to_numeric(df[name], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "field": name, "count": int(s.count()),
            "min": float(s.min()), "p5": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)), "median": float(s.median()),
            "mean": float(s.mean()), "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)), "max": float(s.max()),
            "std": float(s.std()),
        })
    return pd.DataFrame(rows).set_index("field")


def _monthly_mortgage_rates() -> pd.DataFrame:
    weekly = pd.read_csv(FRED_CSV_URL)
    weekly.columns = ["date", "rate"]
    weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")
    weekly["rate"] = pd.to_numeric(weekly["rate"], errors="coerce")
    monthly = (
        weekly.dropna()
        .set_index("date")["rate"]
        .resample("MS")
        .mean()
        .rename("rate_30yr_fixed")
        .reset_index()
    )
    monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)
    print(f"  FRED MORTGAGE30US: {len(monthly)} monthly averages "
          f"({monthly['year_month'].min()} .. {monthly['year_month'].max()})")
    return monthly[["year_month", "rate_30yr_fixed"]]


def _attach_rates(df: pd.DataFrame, date_col: str,
                  rates: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year_month"] = (
        pd.to_datetime(out[date_col], errors="coerce").dt.to_period("M").astype(str)
    )
    out = out.merge(rates, on="year_month", how="left")
    miss = int(out["rate_30yr_fixed"].isna().sum())
    print(f"  rate merge on {date_col}: {miss:,} rows without a rate")
    return out


def stage_eda() -> None:
    _banner("Stage 2/6  profile the data + attach mortgage rates")
    sold = _read(os.path.join(WORK_DIR, "combined_sold.csv"))
    listed = _read(os.path.join(WORK_DIR, "combined_listed.csv"))

    for label, df in (("sold", sold), ("listed", listed)):
        tab = _missing_table(df)
        _save(tab.reset_index().rename(columns={"index": "column"}),
              os.path.join(OUT_DIR, f"{label}_missing_profile.csv"))
        high = tab.index[tab["null_pct"] > 90].tolist()
        print(f"  {label}: {len(high)} columns are >90% empty")

    dist = _distribution_table(sold)
    _save(dist.reset_index(), os.path.join(OUT_DIR, "sold_distribution.csv"))
    cp = pd.to_numeric(sold["ClosePrice"], errors="coerce").dropna()
    dom = pd.to_numeric(sold["DaysOnMarket"], errors="coerce").dropna()

    rates = _monthly_mortgage_rates()
    sold_r = _attach_rates(sold, "CloseDate", rates)
    listed_r = _attach_rates(listed, "ListingContractDate", rates)
    # save only the rate window the data actually spans (FRED history goes
    # back to 1971 — keeping all of it just bloats the output file)
    lo = min(sold_r["year_month"].min(), listed_r["year_month"].min())
    hi = max(sold_r["year_month"].max(), listed_r["year_month"].max())
    window = rates[(rates["year_month"] >= lo) & (rates["year_month"] <= hi)]
    print(f"  saving rate window {lo} .. {hi} ({len(window)} months)")
    _save(window, os.path.join(OUT_DIR, "mortgage_monthly_avg.csv"))
    _save(sold_r, os.path.join(WORK_DIR, "sold_with_rates.csv"))
    _save(listed_r, os.path.join(WORK_DIR, "listed_with_rates.csv"))
    _record("eda", {
        "sold_median_close_price": float(cp.median()),
        "sold_mean_close_price": float(cp.mean()),
        "sold_median_days_on_market": float(dom.median()),
    })


# --------------------------------------------------------------------------
# stage 3 — cleaning, phase 1
# --------------------------------------------------------------------------

NULL_DROP_THRESHOLD = 0.90          # drop a column when >90% of it is empty
PRICE_CEILING = 50_000_000          # prices above $50M are treated as errors
IQR_K = 1.5                         # outlier fences: Q1 - K*IQR / Q3 + K*IQR


def _prune_sparse_columns(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list]:
    sparse = df.columns[df.isnull().mean() > NULL_DROP_THRESHOLD].tolist()
    print(f"  {label}: dropping {len(sparse)} sparse columns {sparse}")
    return df.drop(columns=sparse).copy(), sparse


def _timeline_flags(sold: pd.DataFrame) -> list[str]:
    """Three ways the listing -> contract -> close timeline can contradict itself."""
    made = []
    specs = [
        ("listing_after_close_flag", "ListingContractDate", "CloseDate"),
        ("purchase_after_close_flag", "PurchaseContractDate", "CloseDate"),
        ("negative_timeline_flag", "ListingContractDate", "PurchaseContractDate"),
    ]
    for flag, early, late in specs:
        if {early, late} <= set(sold.columns):
            mask = sold[early].notna() & sold[late].notna() & (sold[early] > sold[late])
            sold[flag] = mask.astype(int)
            made.append(flag)
            print(f"  {flag}: {int(mask.sum()):,} records")
    return made


def _value_flags(df: pd.DataFrame, label: str, rules: list) -> list[str]:
    made = []
    for column, bad, description in rules:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        flag = f"{column}_invalid_flag"
        # several rules can target one column; keep a single combined flag
        hit = bad(values).fillna(False)
        df[flag] = (hit | df.get(flag, 0).astype(bool)).astype(int) \
            if flag in df.columns else hit.astype(int)
        if flag not in made:
            made.append(flag)
        print(f"  {label} {column} ({description}): {int(hit.sum()):,} records")
    return made


def _geo_flags(df: pd.DataFrame, label: str) -> dict:
    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    stats = {
        f"{label}_geo_null_pct": round(float((lat.isna() | lon.isna()).mean() * 100), 2),
        f"{label}_lon_positive": int((lon > 0).sum()),   # California is west: lon < 0
    }
    df["geo_null_flag"] = (lat.isna() | lon.isna()).astype(int)
    df["geo_sentinel_flag"] = ((lat == 0) | (lon == 0)).astype(int)
    df["geo_invalid_flag"] = (lon > 0).astype(int)
    print(f"  {label}: geo null {stats[f'{label}_geo_null_pct']}% | "
          f"lon>0 {stats[f'{label}_lon_positive']:,}")
    return stats


def stage_cleaning() -> None:
    _banner("Stage 3/6  cleaning phase 1 — prune, parse, flag")
    sold = _read(os.path.join(WORK_DIR, "sold_with_rates.csv"))
    listed = _read(os.path.join(WORK_DIR, "listed_with_rates.csv"))

    sold, sold_dropped = _prune_sparse_columns(sold, "sold")
    listed, listed_dropped = _prune_sparse_columns(listed, "listed")

    for df, cols in (
        (sold, ["ListingContractDate", "PurchaseContractDate", "CloseDate",
                "StatusChangeTimestamp", "ModificationTimestamp"]),
        (listed, ["ListingContractDate", "ExpirationDate",
                  "StatusChangeTimestamp", "ModificationTimestamp"]),
    ):
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    timeline = _timeline_flags(sold)
    sold_flags = _value_flags(sold, "sold", [
        ("ClosePrice", lambda s: s <= 0, "<= 0"),
        ("ClosePrice", lambda s: s > PRICE_CEILING, "> $50M"),
        ("ListPrice", lambda s: s <= 0, "<= 0"),
        ("OriginalListPrice", lambda s: s <= 0, "<= 0"),
        ("OriginalListPrice", lambda s: s > PRICE_CEILING, "> $50M"),
        ("LivingArea", lambda s: s <= 0, "<= 0"),
        ("DaysOnMarket", lambda s: s < 0, "< 0"),
    ])
    listed_flags = _value_flags(listed, "listed", [
        ("ListPrice", lambda s: s <= 0, "<= 0"),
        ("ListPrice", lambda s: s > PRICE_CEILING, "> $50M"),
        ("LivingArea", lambda s: s <= 0, "<= 0"),
    ])

    geo_stats = {}
    geo_stats.update(_geo_flags(sold, "sold"))
    geo_stats.update(_geo_flags(listed, "listed"))

    for df, flags in ((sold, sold_flags + timeline), (listed, listed_flags)):
        present = [f for f in flags if f in df.columns]
        df["any_invalid_flag"] = df[present].max(axis=1).astype(int)

    _save(sold, os.path.join(WORK_DIR, "sold_clean_flagged.csv"))
    _save(listed, os.path.join(WORK_DIR, "listed_clean_flagged.csv"))
    _record("cleaning_phase1", {
        "sold_dropped_columns": sold_dropped,
        "listed_dropped_columns": listed_dropped,
        "sold_any_invalid": int(sold["any_invalid_flag"].sum()),
        "listed_any_invalid": int(listed["any_invalid_flag"].sum()),
        **geo_stats,
    })


# --------------------------------------------------------------------------
# stage 4 — cleaning, phase 2 (the step the shared repo never included;
#           rebuilt here from the project specification)
# --------------------------------------------------------------------------

CA_LAT_RANGE = (32.5, 42.0)         # California bounding box, handbook spec
CA_LON_RANGE = (-124.4, -114.1)


def _finalise(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    inside = (
        lat.between(*CA_LAT_RANGE) & lon.between(*CA_LON_RANGE)
    )
    has_geo = lat.notna() & lon.notna()
    df["geo_outside_ca_flag"] = (has_geo & ~inside).astype(int)
    n_out = int(df["geo_outside_ca_flag"].sum())
    print(f"  {label}: {n_out:,} records with coordinates outside California")

    audit = {
        "numeric_columns_checked": int(
            sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)),
        "datetime_columns_checked": int(
            sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)),
    }
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    df["flag_total"] = df[flag_cols].sum(axis=1)
    facts = {
        f"{label}_outside_ca": n_out,
        f"{label}_any_flag_records": int((df["flag_total"] > 0).sum()),
        f"{label}_dtype_audit": audit,
    }
    return df, facts


def stage_geoclean() -> None:
    _banner("Stage 4/6  cleaning phase 2 — California bounds + dtype audit")
    sold = _read(os.path.join(WORK_DIR, "sold_clean_flagged.csv"))
    listed = _read(os.path.join(WORK_DIR, "listed_clean_flagged.csv"))
    # dates come back from CSV as text — parse them before the dtype audit
    for df in (sold, listed):
        for col in ("ListingContractDate", "PurchaseContractDate", "CloseDate",
                    "ExpirationDate", "StatusChangeTimestamp",
                    "ModificationTimestamp", "year_month"):
            if col in df.columns and col != "year_month":
                df[col] = pd.to_datetime(df[col], errors="coerce")
    sold, sold_facts = _finalise(sold, "sold")
    listed, listed_facts = _finalise(listed, "listed")
    _save(sold, os.path.join(WORK_DIR, "sold_final_clean.csv"))
    _save(listed, os.path.join(WORK_DIR, "listed_final_clean.csv"))
    _record("cleaning_phase2", {**sold_facts, **listed_facts})


# --------------------------------------------------------------------------
# stage 5 — features and segments (Sold only, as in the spec)
# --------------------------------------------------------------------------


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def stage_features() -> None:
    _banner("Stage 5/6  feature engineering + segment tables")
    sold = _read(os.path.join(WORK_DIR, "sold_final_clean.csv"))
    for col in ("ClosePrice", "OriginalListPrice", "ListPrice", "LivingArea",
                "LotSizeAcres", "DaysOnMarket", "BedroomsTotal",
                "BathroomsTotalInteger", "YearBuilt"):
        if col in sold.columns:
            sold[col] = pd.to_numeric(sold[col], errors="coerce")
    for col in ("ListingContractDate", "PurchaseContractDate", "CloseDate"):
        sold[col] = pd.to_datetime(sold[col], errors="coerce")

    sold["price_ratio"] = _safe_ratio(sold["ClosePrice"], sold["OriginalListPrice"])
    sold["close_to_orig_list_ratio"] = sold["price_ratio"]
    sold["ppsf"] = _safe_ratio(sold["ClosePrice"], sold["LivingArea"])
    sold["close_year"] = sold["CloseDate"].dt.year
    sold["close_month"] = sold["CloseDate"].dt.month
    sold["yr_mo"] = sold["CloseDate"].dt.to_period("M").astype(str)
    sold["listing_to_contract_days"] = (
        sold["PurchaseContractDate"] - sold["ListingContractDate"]).dt.days
    sold["contract_to_close_days"] = (
        sold["CloseDate"] - sold["PurchaseContractDate"]).dt.days

    sold["price_ratio_flag"] = (
        (sold["price_ratio"] < 0.3) | (sold["price_ratio"] > 2.0)).astype(int)
    sold["ppsf_flag"] = ((sold["ppsf"] < 10) | (sold["ppsf"] > 5000)).astype(int)
    sold["ltc_days_flag"] = (
        (sold["listing_to_contract_days"] < 0)
        | (sold["listing_to_contract_days"] > 1825)).astype(int)
    sold["ctc_days_flag"] = (
        (sold["contract_to_close_days"] < 0)
        | (sold["contract_to_close_days"] > 365)).astype(int)

    usable = sold.loc[
        sold["price_ratio"].notna() & sold["ppsf"].notna()
        & (sold["ClosePrice"] > 0) & (sold["ppsf_flag"] == 0)
    ].copy()
    print(f"  segment-ready records: {len(usable):,} of {len(sold):,}")

    agg = {
        "ClosePrice": ["count", "median", "mean"],
        "ppsf": ["median", "mean"],
        "price_ratio": ["median", "mean"],
        "DaysOnMarket": ["median", "mean"],
        "listing_to_contract_days": "median",
        "contract_to_close_days": "median",
    }

    def segment(frame: pd.DataFrame, key: str, top: int | None = None) -> pd.DataFrame:
        table = frame.groupby(frame[key].fillna("Unknown")).agg(agg).round(2)
        table.columns = ["_".join(c).strip("_") for c in table.columns]
        table = table.reset_index()
        table = table[table["ClosePrice_count"] >= 10]
        table = table.sort_values("ClosePrice_count", ascending=False)
        return table.head(top) if top else table

    _save(segment(usable, "CountyOrParish"),
          os.path.join(OUT_DIR, "segment_by_county.csv"))
    _save(segment(usable, "PropertySubType"),
          os.path.join(OUT_DIR, "segment_by_subtype.csv"))
    _save(segment(usable, "ListOfficeName", top=20),
          os.path.join(OUT_DIR, "segment_by_office.csv"))

    monthly = (
        usable.groupby("yr_mo")
        .agg(ClosePrice_count=("ClosePrice", "count"),
             ClosePrice_median=("ClosePrice", "median"),
             ppsf_median=("ppsf", "median"),
             price_ratio_median=("price_ratio", "median"),
             DaysOnMarket_median=("DaysOnMarket", "median"))
        .round(2)
        .reset_index()
        .sort_values("yr_mo")
    )
    _save(monthly, os.path.join(OUT_DIR, "monthly_market_trend.csv"))
    _save(sold, os.path.join(WORK_DIR, "sold_with_features.csv"))
    _record("features", {
        "ppsf_valid_pct": round(float(sold["ppsf"].notna().mean() * 100), 2),
        "price_ratio_valid_pct": round(float(sold["price_ratio"].notna().mean() * 100), 2),
        "counties": int(usable["CountyOrParish"].nunique()),
        "months_in_trend": int(monthly.shape[0]),
        # segment/trend tables describe the feature-valid population BEFORE
        # outlier exclusion; analysis-ready is a different, smaller
        # population. The two must never be mixed without this label.
        "segment_population_rows": int(len(usable)),
        "segment_population": "feature-valid rows before outlier exclusion",
    })


# --------------------------------------------------------------------------
# stage 6 — outlier detection and the analysis-ready dataset
# --------------------------------------------------------------------------

IQR_FIELDS = [
    "ClosePrice", "LivingArea", "DaysOnMarket", "ppsf",
    "listing_to_contract_days", "contract_to_close_days",
]


def stage_outliers() -> None:
    _banner("Stage 6/6  business rules + IQR outlier detection")
    sold = _read(os.path.join(WORK_DIR, "sold_with_features.csv"))
    for col in set(IQR_FIELDS + ["OriginalListPrice", "ListPrice"]):
        if col in sold.columns:
            sold[col] = pd.to_numeric(sold[col], errors="coerce")

    # business rules first: these records are invalid on their face
    sold["biz_close_price_invalid"] = (
        sold["ClosePrice"].isna() | (sold["ClosePrice"] <= 0)).astype(int)
    sold["biz_living_area_invalid"] = (
        sold["LivingArea"].isna() | (sold["LivingArea"] <= 0)).astype(int)
    sold["biz_dom_invalid"] = (sold["DaysOnMarket"] < 0).astype(int)
    biz_cols = ["biz_close_price_invalid", "biz_living_area_invalid", "biz_dom_invalid"]
    sold["any_biz_invalid"] = sold[biz_cols].max(axis=1)
    base = sold.loc[sold["any_biz_invalid"] == 0]
    print(f"  business-rule invalids: {int(sold['any_biz_invalid'].sum()):,} | "
          f"IQR base: {len(base):,}")

    stats, flag_cols = {}, []
    for col in IQR_FIELDS:
        series = base[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        flag = f"{col}_outlier_flag"
        hit = sold[col].notna() & (sold["any_biz_invalid"] == 0) & (
            (sold[col] < low) | (sold[col] > high))
        sold[flag] = hit.astype(int)
        flag_cols.append(flag)
        stats[col] = {
            "q1": round(float(q1), 2), "q3": round(float(q3), 2),
            "lower": round(float(low), 2), "upper": round(float(high), 2),
            "flagged": int(hit.sum()),
        }
        print(f"  {col:28s} bounds [{low:>12,.0f}, {high:>12,.0f}]  "
              f"flagged {int(hit.sum()):,}")

    sold["any_outlier_flag"] = sold[flag_cols].max(axis=1).astype(int)
    sold["exclude_flag"] = (
        (sold["any_biz_invalid"] == 1) | (sold["any_outlier_flag"] == 1)).astype(int)

    # ---- final data-quality gate (extension control layer) -------------
    # The cleaning stages follow the original spec: they FLAG suspect
    # records for audit and keep them in the flow. A production deliverable
    # needs the opposite at the boundary: nothing ships with a known data
    # defect. The gate therefore hard-excludes any surviving record that
    # still carries a validity flag — coordinates outside California or
    # invalid, $/sqft or sale-to-list outside their validity ranges,
    # timeline spans outside theirs, or any phase-1 invalid flag. Missing
    # coordinates (geo_null) are NOT gated: absence of information is not
    # an error, and those rows stay, visibly flagged. Every gated row is
    # attributed to one primary reason (first matching flag, in the order
    # below) so the reconciliation bridge can account for it exactly.
    GATE_FLAGS = ["geo_outside_ca_flag", "geo_invalid_flag", "ppsf_flag",
                  "price_ratio_flag", "ltc_days_flag", "ctc_days_flag",
                  "any_invalid_flag"]
    present = [c for c in GATE_FLAGS if c in sold.columns]
    cand = sold.loc[sold["exclude_flag"] == 0]
    gate_hit = cand[present].max(axis=1).astype(int) if present \
        else pd.Series(0, index=cand.index)
    sold["gate_flag"] = 0
    sold.loc[cand.index, "gate_flag"] = gate_hit
    primary = {}
    remaining = cand.loc[gate_hit == 1]
    for col in present:
        hit = remaining.loc[remaining[col] == 1]
        if len(hit):
            primary[col] = int(len(hit))
            remaining = remaining.drop(index=hit.index)
    gate_by_flag = {c: int(cand[c].sum()) for c in present}
    sold["exclude_flag"] = (
        (sold["exclude_flag"] == 1) | (sold["gate_flag"] == 1)).astype(int)
    ready = sold.loc[sold["exclude_flag"] == 0].copy()
    n_gate = int(gate_hit.sum())
    print(f"  final DQ gate: excluded {n_gate:,} flagged survivors "
          f"(primary reasons: {primary})")

    med_before = float(sold["ClosePrice"].median())
    med_after = float(ready["ClosePrice"].median())
    print(f"  excluded {int(sold['exclude_flag'].sum()):,} | "
          f"analysis-ready {len(ready):,} | median ClosePrice "
          f"${med_before:,.0f} -> ${med_after:,.0f}")

    _save(sold, os.path.join(WORK_DIR, "sold_outlier_flagged.csv"))
    _save(ready, os.path.join(OUT_DIR, "sold_analysis_ready.csv"))
    _record("outliers", {
        "rows_before": len(sold), "rows_analysis_ready": len(ready),
        "rows_excluded": int(sold["exclude_flag"].sum()),
        "median_close_before": med_before, "median_close_after": med_after,
        "iqr": stats,
        "gate": {"rows_excluded": n_gate, "by_flag": gate_by_flag,
                 "by_primary_reason": primary,
                 "policy": "hard-exclude surviving records carrying a "
                           "validity flag; missing coordinates are "
                           "flagged, not gated"},
    })


STAGES = {
    "1": stage_aggregate, "2": stage_eda, "3": stage_cleaning,
    "4": stage_geoclean, "5": stage_features, "6": stage_outliers,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("stages", nargs="+",
                        help="'all' or stage numbers, e.g. 1 2 3")
    args = parser.parse_args(argv)
    wanted = list(STAGES) if args.stages == ["all"] else args.stages
    _ensure_dirs()
    print(f"IDX pipeline (independent reproduction) — author: {AUTHOR}")
    import time
    started = time.perf_counter()
    for key in wanted:
        tick = time.perf_counter()
        STAGES[key]()
        print(f"  [stage {key} took {time.perf_counter() - tick:.1f}s]")
    total = round(time.perf_counter() - started, 1)
    _record("_runtime", {"stages_run": wanted, "seconds": total})
    print(f"\nDone in {total}s. This run's facts: output/run_facts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
