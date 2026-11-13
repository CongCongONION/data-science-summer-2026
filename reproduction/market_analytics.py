"""
market_analytics.py — market structure analytics beyond the base segments

Author : Anna

Run AFTER idx_pipeline.py. The base pipeline produces segment medians and a
monthly trend; this module turns the same run's outputs into the analytics
a housing-market reviewer expects:

  * MONTHLY MARKET INDEX — median price indexed to the first quarter of
    the sample (= 100), with median $/sqft, days-on-market, sale-to-list
    ratio, sales volume, and the contemporaneous 30-yr mortgage rate on
    one monthly spine.
  * RATE OVERLAY — Pearson and Spearman correlations between the monthly
    mortgage rate and volume / price / liquidity (DOM). Reported as
    measured associations with their sample size — not causal claims.
  * COUNTY DEEP DIVE — per-county volume share, price level and spread
    (IQR), $/sqft, liquidity and sale-to-list, so segment comparisons
    carry dispersion, not just a median.
  * SUPPLY SIDE — monthly new-listing counts from the Listed feed against
    monthly closings, a first-pass tightness gauge.

Outputs (output/): monthly_market_index.csv, county_deep_dive.csv,
analytics_facts.json

  python market_analytics.py
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


def monthly_index(ready: pd.DataFrame) -> pd.DataFrame:
    g = ready.groupby("yr_mo").agg(
        sales=("ClosePrice", "size"),
        median_price=("ClosePrice", "median"),
        median_ppsf=("ppsf", "median"),
        median_dom=("DaysOnMarket", "median"),
        median_sale_to_list=("price_ratio", "median"),
        mean_rate=("rate_30yr_fixed", "mean")).reset_index()
    base = g.loc[g["sales"] >= 10, "median_price"].head(3).mean()
    g["price_index"] = (100 * g["median_price"] / base).round(1)
    return g


def rate_overlay(idx: pd.DataFrame) -> dict:
    solid = idx.loc[idx["sales"] >= 10]  # thin tail months add noise only
    out = {"months_used": int(len(solid))}
    for col in ("sales", "median_price", "median_dom"):
        # Spearman computed as Pearson on ranks — identical definition,
        # without taking a scipy dependency for one coefficient
        out[f"rate_vs_{col}"] = {
            "pearson": round(float(solid["mean_rate"].corr(solid[col])), 3),
            "spearman": round(float(solid["mean_rate"].rank().corr(
                solid[col].rank())), 3),
        }
    return out


def county_dive(ready: pd.DataFrame) -> pd.DataFrame:
    g = ready.groupby("CountyOrParish").agg(
        sales=("ClosePrice", "size"),
        median_price=("ClosePrice", "median"),
        p25_price=("ClosePrice", lambda s: s.quantile(0.25)),
        p75_price=("ClosePrice", lambda s: s.quantile(0.75)),
        median_ppsf=("ppsf", "median"),
        median_dom=("DaysOnMarket", "median"),
        median_sale_to_list=("price_ratio", "median")).reset_index()
    g["volume_share_pct"] = (100 * g["sales"] / g["sales"].sum()).round(1)
    g["price_iqr"] = g["p75_price"] - g["p25_price"]
    return g.sort_values("sales", ascending=False)


def supply_side() -> dict:
    listed = _read(os.path.join(WORK, "combined_listed.csv"))
    listed["yr_mo"] = pd.to_datetime(
        listed["ListingContractDate"], errors="coerce").dt.to_period("M")
    new_listings = listed.groupby("yr_mo").size()
    return {"listed_rows": int(len(listed)),
            "peak_listing_month": str(new_listings.idxmax()),
            "peak_new_listings": int(new_listings.max()),
            "median_monthly_new_listings": int(new_listings.median())}


def main() -> None:
    ready = _read(os.path.join(OUT, "sold_analysis_ready.csv"))
    idx = monthly_index(ready)
    idx.to_csv(os.path.join(OUT, "monthly_market_index.csv"), index=False)
    dive = county_dive(ready)
    dive.to_csv(os.path.join(OUT, "county_deep_dive.csv"), index=False)
    facts = {
        "index_base": "median price, first 3 months with >= 10 sales = 100",
        "months": int(len(idx)),
        "latest_index": float(idx["price_index"].iloc[-1]),
        "rate_overlay": rate_overlay(idx),
        "supply": supply_side(),
        "county_leader_by_volume": str(dive.iloc[0]["CountyOrParish"]),
        "county_price_leader": str(
            dive.loc[dive["median_price"].idxmax(), "CountyOrParish"]),
    }
    with open(os.path.join(OUT, "analytics_facts.json"), "w",
              encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2)
    ro = facts["rate_overlay"]
    print(f"  monthly index: {facts['months']} months, latest "
          f"{facts['latest_index']:.1f} (base = 100)")
    print(f"  rate vs volume: Pearson {ro['rate_vs_sales']['pearson']:+.2f} "
          f"/ Spearman {ro['rate_vs_sales']['spearman']:+.2f} "
          f"({ro['months_used']} solid months)")
    print(f"  volume leader: {facts['county_leader_by_volume']} | "
          f"price leader: {facts['county_price_leader']}")
    print("  wrote output/monthly_market_index.csv, "
          "output/county_deep_dive.csv, output/analytics_facts.json")


if __name__ == "__main__":
    main()
