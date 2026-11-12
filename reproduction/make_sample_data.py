"""
make_sample_data.py — synthetic stand-in for the CRMLS monthly exports

Author : Anna
Date   : November 2026

The real CRMLS exports are licensed data and are not in this repo, so this
generator creates small monthly Sold / Listing CSVs with the same schema,
roughly the same distributions, and the same kinds of defects the cleaning
stages exist to catch (a few zero living areas, negative days-on-market,
contradictory timelines, coordinates outside California, sparse columns).

It exists ONLY so the pipeline can be executed end-to-end and verified.
Swap the contents of data/raw/ for the real exports and re-run
idx_pipeline.py to get real results — the pipeline does not care which
files are in those folders.

  python make_sample_data.py            # writes data/raw/Sold + data/raw/Listing
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261110)

ROOT = os.path.dirname(os.path.abspath(__file__))
MONTHS = pd.period_range("2024-01", "2026-03", freq="M")

COUNTIES = {
    "Los Angeles": (34.05, -118.24), "Orange": (33.72, -117.83),
    "San Diego": (32.72, -117.16), "Riverside": (33.95, -117.40),
    "San Bernardino": (34.11, -117.29), "Ventura": (34.28, -119.29),
}
SUBTYPES = ["Single Family Residence", "Condominium", "Townhouse",
            "Manufactured Home", "Residential Income"]
OFFICES = [f"Office {letter}" for letter in "ABCDEFGHIJ"]


def _batch(n: int, month: pd.Period, sold: bool) -> pd.DataFrame:
    county_names = list(COUNTIES)
    county = rng.choice(county_names, size=n, p=[.34, .22, .18, .12, .09, .05])
    base_lat = np.array([COUNTIES[c][0] for c in county])
    base_lon = np.array([COUNTIES[c][1] for c in county])
    lat = base_lat + rng.normal(0, 0.15, n)
    lon = base_lon + rng.normal(0, 0.15, n)

    living = np.clip(rng.lognormal(7.5, 0.45, n), 300, 9000)
    ppsf = rng.lognormal(6.15, 0.28, n)                     # ~ $470 / sq ft
    close_price = (living * ppsf).round(-3)
    orig_list = (close_price / rng.normal(0.99, 0.04, n)).round(-3)
    list_price = (orig_list * rng.normal(0.99, 0.02, n)).round(-3)

    list_day = pd.Timestamp(month.start_time) + pd.to_timedelta(
        rng.integers(0, 27, n), unit="D")
    dom = np.clip(rng.gamma(2.0, 18.0, n), 0, 400).astype(int)
    contract_day = list_day + pd.to_timedelta(dom, unit="D")
    escrow = np.clip(rng.normal(38, 10, n), 10, 120).astype(int)
    close_day = contract_day + pd.to_timedelta(escrow, unit="D")

    df = pd.DataFrame({
        "ListingId": [f"S{month}{i:05d}" if sold else f"L{month}{i:05d}"
                      for i in range(n)],
        "PropertyType": rng.choice(
            ["Residential", "Land", "Commercial"], size=n, p=[.93, .04, .03]),
        "PropertySubType": rng.choice(SUBTYPES, size=n),
        "CountyOrParish": county,
        "City": "Sample City",
        "StateOrProvince": "CA",
        "PostalCode": "90001",
        "Latitude": lat.round(6),
        "Longitude": lon.round(6),
        "ClosePrice": close_price if sold else np.nan,
        "ListPrice": list_price,
        "OriginalListPrice": orig_list,
        "LivingArea": living.round(0),
        "LotSizeAcres": np.clip(rng.lognormal(-1.6, 0.7, n), 0.02, 5).round(3),
        "BedroomsTotal": rng.integers(1, 7, n),
        "BathroomsTotalInteger": rng.integers(1, 6, n),
        "DaysOnMarket": dom,
        "YearBuilt": rng.integers(1920, 2025, n),
        "ListOfficeName": rng.choice(OFFICES, size=n),
        "ListingContractDate": list_day.strftime("%Y-%m-%d"),
        "PurchaseContractDate": contract_day.strftime("%Y-%m-%d") if sold else "",
        "CloseDate": close_day.strftime("%Y-%m-%d") if sold else "",
        "ExpirationDate": "" if sold else (list_day + pd.Timedelta(days=180)).strftime("%Y-%m-%d"),
        "StatusChangeTimestamp": close_day.strftime("%Y-%m-%d %H:%M:%S") if sold else list_day.strftime("%Y-%m-%d %H:%M:%S"),
        "ModificationTimestamp": list_day.strftime("%Y-%m-%d %H:%M:%S"),
        # near-empty columns, like the real exports carry
        "VirtualTourURLUnbranded": [None] * n,
        "FireplaceFeatures": [None if rng.random() > 0.03 else "Yes" for _ in range(n)],
    })

    # inject the defects the cleaners are built to catch (~0.4-1% each)
    def spoil(mask_p, fn):
        idx = rng.random(n) < mask_p
        if idx.any():
            fn(df, idx)

    spoil(0.006, lambda d, i: d.__setitem__("LivingArea", np.where(i, 0, d["LivingArea"])))
    spoil(0.005, lambda d, i: d.__setitem__("DaysOnMarket", np.where(i, -3, d["DaysOnMarket"])))
    spoil(0.04, lambda d, i: d.__setitem__("Latitude", np.where(i, np.nan, d["Latitude"])))
    spoil(0.004, lambda d, i: d.__setitem__("Longitude", np.where(i, 118.2, d["Longitude"])))  # wrong hemisphere
    if sold:
        spoil(0.01, lambda d, i: d.__setitem__("ClosePrice", d["ClosePrice"] * np.where(i, 25.0, 1.0)))  # price outliers
        spoil(0.005, lambda d, i: d.__setitem__("CloseDate", np.where(i, d["ListingContractDate"], d["CloseDate"])))
    return df


def main() -> None:
    marker_dir = os.path.join(ROOT, "data", "raw")
    os.makedirs(marker_dir, exist_ok=True)
    with open(os.path.join(marker_dir, "SAMPLE_DATA.txt"), "w") as fh:
        fh.write("These CSVs are synthetic sample data generated by "
                 "make_sample_data.py (CRMLS schema, seeded RNG).\n"
                 "Delete this file when real CRMLS exports replace them.\n")
    for kind, per_month, sold in (("Sold", 260, True), ("Listing", 340, False)):
        folder = os.path.join(ROOT, "data", "raw", kind)
        os.makedirs(folder, exist_ok=True)
        total = 0
        for month in MONTHS:
            frame = _batch(per_month, month, sold)
            frame.to_csv(os.path.join(folder, f"CRMLS{kind}{month.strftime('%Y%m')}.csv"),
                         index=False)
            total += len(frame)
        print(f"{kind}: {len(MONTHS)} monthly files, {total:,} rows -> data/raw/{kind}/")
    print("Sample data ready. Now run:  python idx_pipeline.py all")


if __name__ == "__main__":
    main()
