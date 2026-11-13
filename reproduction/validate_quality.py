"""
validate_quality.py — data-quality scorecard + row-count reconciliation

Author : Anna

Run AFTER idx_pipeline.py. Two professional controls the base pipeline
implies but never states explicitly:

1. RECONCILIATION BRIDGE — every row that enters the pipeline is accounted
   for at every stage boundary: combined -> cleaned -> features ->
   analysis-ready, with each delta attributed to a named reason (residential
   filter is applied at ingest; exclusions split into business-rule invalids
   vs statistical outliers). The bridge must close exactly; if it does not,
   this script fails loudly instead of writing a report over a leak.

2. DATA-QUALITY SCORECARD — named checks with explicit thresholds over the
   analysis-ready dataset and its inputs: completeness of key fields,
   validity ranges, uniqueness of the listing key, timeline consistency,
   geographic validity, and mortgage-rate merge coverage. Each check reports
   its measured value, its threshold, and PASS / WARN / FAIL, so a reviewer
   can see the standard the data was held to — not just the outcome.

Outputs (output/):
  row_count_bridge.csv        one row per stage boundary, deltas attributed
  data_quality_scorecard.csv  one row per check
  quality_facts.json          machine-readable copy for the report/tests

  python validate_quality.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "data", "work")
OUT = os.path.join(ROOT, "output")


def _read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def build_bridge() -> pd.DataFrame:
    """Account for every Sold row from ingest to analysis-ready."""
    combined = _read(os.path.join(WORK, "combined_sold.csv"))
    final_clean = _read(os.path.join(WORK, "sold_final_clean.csv"))
    features = _read(os.path.join(WORK, "sold_with_features.csv"))
    flagged = _read(os.path.join(WORK, "sold_outlier_flagged.csv"))
    ready = _read(os.path.join(OUT, "sold_analysis_ready.csv"))

    biz = int(flagged["any_biz_invalid"].sum())
    out_only = int(((flagged["any_outlier_flag"] == 1)
                    & (flagged["any_biz_invalid"] == 0)).sum())
    gated = int(flagged["gate_flag"].sum()) if "gate_flag" in flagged \
        else 0
    rows = [
        ("combined_sold (Residential, as ingested)", len(combined), 0, ""),
        ("after cleaning phases 1-2 (flags added, no rows dropped)",
         len(final_clean), len(final_clean) - len(combined),
         "cleaning flags records, it does not remove them"),
        ("after feature engineering", len(features),
         len(features) - len(final_clean), "features add columns only"),
        ("excluded: business-rule invalids", -biz, -biz,
         "ClosePrice/LivingArea missing or <= 0, or DaysOnMarket < 0"),
        ("excluded: statistical outliers (1.5x IQR, biz-valid rows only)",
         -out_only, -out_only,
         "outside fences on price, area, DOM, $/sqft or timeline spans"),
        ("excluded: final data-quality gate", -gated, -gated,
         "survivors still carrying a validity flag (geo outside CA or "
         "invalid, feature ranges, timeline spans)"),
        ("analysis-ready", len(ready), 0, "sold_analysis_ready.csv"),
    ]
    bridge = pd.DataFrame(
        rows, columns=["stage", "rows", "delta_vs_previous", "note"])
    accounted = len(features) - biz - out_only - gated
    assert accounted == len(ready), (
        f"BRIDGE DOES NOT CLOSE: features {len(features):,} - biz {biz:,} "
        f"- outliers {out_only:,} - gate {gated:,} = {accounted:,} "
        f"!= ready {len(ready):,}")
    assert len(combined) == len(final_clean) == len(features) \
        == len(flagged), "row count changed before the exclusion stage"
    print(f"  bridge closes: {len(features):,} = {biz:,} biz-invalid + "
          f"{out_only:,} outliers + {gated:,} gated + "
          f"{len(ready):,} analysis-ready")
    return bridge


def _check(name, value, threshold, status, detail):
    return {"check": name, "value": value, "threshold": threshold,
            "status": status, "detail": detail}


def build_scorecard() -> pd.DataFrame:
    ready = _read(os.path.join(OUT, "sold_analysis_ready.csv"))
    combined = _read(os.path.join(WORK, "combined_sold.csv"))
    listed = _read(os.path.join(WORK, "combined_listed.csv"))
    n = len(ready)
    checks = []

    def pct(mask: pd.Series) -> float:
        return round(100.0 * float(mask.mean()), 2)

    for col in ("ClosePrice", "LivingArea", "CloseDate", "CountyOrParish"):
        v = pct(ready[col].notna())
        checks.append(_check(f"completeness: {col} present (Sold, ready)",
                             v, ">= 99%", "PASS" if v >= 99 else "FAIL",
                             f"{int(ready[col].notna().sum()):,} of {n:,} rows"))
    v = pct((ready["ClosePrice"] > 0) & (ready["ClosePrice"] <= 50_000_000))
    checks.append(_check("validity: ClosePrice in (0, $50M]", v, "== 100%",
                         "PASS" if v == 100 else "FAIL",
                         "business-rule ceiling from the pipeline spec"))
    v = pct(ready["ppsf"].between(10, 5000))
    checks.append(_check("validity: $/sqft within [10, 5000]", v, ">= 99%",
                         "PASS" if v >= 99 else "WARN",
                         "feature-validity range, Week 6 spec"))
    v = pct(ready["price_ratio"].between(0.3, 2.0))
    checks.append(_check("validity: sale-to-list ratio within [0.3, 2.0]", v,
                         ">= 99%", "PASS" if v >= 99 else "WARN",
                         "feature-validity range, Week 6 spec"))
    dup = int(ready["ListingId"].duplicated().sum())
    checks.append(_check("uniqueness: duplicate ListingId (Sold, ready)", dup,
                         "== 0", "PASS" if dup == 0 else "WARN",
                         "repeat sales share an id only across different "
                         "CloseDates; exact-key duplicates are the concern"))
    dup_full = int(ready.duplicated(
        subset=["ListingId", "CloseDate"]).sum())
    checks.append(_check("uniqueness: duplicate (ListingId, CloseDate)", dup_full,
                         "== 0", "PASS" if dup_full == 0 else "FAIL",
                         "same listing closing twice on one date = double count"))
    dates = ready.assign(
        cd=pd.to_datetime(ready["CloseDate"], errors="coerce"),
        pd_=pd.to_datetime(ready["PurchaseContractDate"], errors="coerce"))
    v = pct(dates["cd"] >= dates["pd_"])
    checks.append(_check("consistency: CloseDate >= PurchaseContractDate", v,
                         "== 100%", "PASS" if v == 100 else "FAIL",
                         "timeline ordering on analysis-ready rows"))
    v = pct(ready["geo_outside_ca_flag"] == 0)
    checks.append(_check("geographic validity: coordinates inside California",
                         v, ">= 99%", "PASS" if v >= 99 else "WARN",
                         "bounding-box audit, Week 5 spec"))
    gate_cols = [c for c in ("geo_outside_ca_flag", "geo_invalid_flag",
                             "ppsf_flag", "price_ratio_flag",
                             "ltc_days_flag", "ctc_days_flag",
                             "any_invalid_flag") if c in ready.columns]
    flagged_rows = int((ready[gate_cols].max(axis=1) > 0).sum()) \
        if gate_cols else 0
    checks.append(_check("final gate: analysis-ready rows carrying a "
                         "validity flag", flagged_rows, "== 0",
                         "PASS" if flagged_rows == 0 else "FAIL",
                         "the gate's own post-condition, checked on the "
                         "shipped file rather than trusted"))
    v = pct(ready["rate_30yr_fixed"].notna())
    checks.append(_check("enrichment: FRED mortgage rate matched", v,
                         "== 100%", "PASS" if v == 100 else "FAIL",
                         "every close month must carry a 30-yr fixed rate"))
    v = pct(listed["ListingId"].notna())
    checks.append(_check("completeness: ListingId present (Listed, combined)",
                         v, ">= 99%", "PASS" if v >= 99 else "FAIL",
                         f"{len(listed):,} listed rows at ingest"))
    v = round(100.0 * float(combined["ClosePrice"].notna().mean()), 2)
    checks.append(_check("completeness: ClosePrice present at ingest (Sold)",
                         v, ">= 95%", "PASS" if v >= 95 else "WARN",
                         "raw-feed completeness before any cleaning"))
    card = pd.DataFrame(checks)
    tally = card["status"].value_counts().to_dict()
    print(f"  scorecard: {tally}")
    return card


def main() -> None:
    print("Validation — reconciliation bridge + data-quality scorecard")
    bridge = build_bridge()
    card = build_scorecard()
    bridge.to_csv(os.path.join(OUT, "row_count_bridge.csv"), index=False)
    card.to_csv(os.path.join(OUT, "data_quality_scorecard.csv"), index=False)
    facts = {
        "bridge_closes": True,
        "bridge_rows": bridge.to_dict("records"),
        "scorecard": card.to_dict("records"),
        "status_tally": card["status"].value_counts().to_dict(),
    }
    with open(os.path.join(OUT, "quality_facts.json"), "w",
              encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("  wrote output/row_count_bridge.csv, "
          "output/data_quality_scorecard.csv, output/quality_facts.json")


if __name__ == "__main__":
    main()
