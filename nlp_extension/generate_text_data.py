"""Generate synthetic PublicRemarks text for the analysis-ready Sold records.

PROVENANCE / HONESTY NOTE
-------------------------
The reproduction dataset is synthetic sample data in the CRMLS schema (the
licensed CRMLS exports are not in any repository). This script goes one step
further: it *generates* MLS-style remark text from the structured fields of
each analysis-ready record, using seeded templates with lexical variation.

The mention labels (condition, tenant occupancy, HOA, flood mention, ...)
are drawn first and the text is then written to express them, so the gold
labels in ``data/gold_labels.csv`` match the text by construction. That makes
extraction and classification scores an UPPER BOUND relative to real-world
remarks, which are messier and only partly labelled. Every report in this
folder states this limitation; the value of the exercise is a working,
evaluated NLP + validation pipeline, not a claim about real CRMLS text.

Deterministic: seed 20261115. Re-running reproduces byte-identical outputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20261115
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "reproduction" / "output" / "sold_analysis_ready.csv"

OPENERS = [
    "Welcome to this {subtype_phrase} in {city}",
    "Located in the heart of {city}, this {subtype_phrase}",
    "Tucked away in {city}, this {subtype_phrase}",
    "Proudly offered in {city}: a {subtype_phrase}",
    "Discover this {subtype_phrase} in {city}",
]
SUBTYPE_PHRASE = {
    "Single Family Residence": ["single family home", "detached home", "family residence"],
    "Condominium": ["condo", "condominium unit", "condo residence"],
    "Townhouse": ["townhome", "townhouse", "townhome residence"],
    "Manufactured Home": ["manufactured home", "mobile home"],
    "Multi Family": ["multi-family property", "income property"],
}
LAYOUT = [
    "featuring {beds} bedrooms and {baths} bathrooms across {sqft} sq ft of living space",
    "offering {beds} bd / {baths} ba and {sqft} square feet of living area",
    "with {beds} beds, {baths} baths, and {sqft} sqft of interior space",
    "boasting {beds} bedrooms, {baths} full baths, and {sqft} SF of living space",
]
YEAR_LOT = [
    "Built in {year} on a {lot} acre lot",
    "Originally constructed in {year}, the home sits on a {lot}-acre parcel",
    "Built {year} with a {lot} acre lot",
]
CONDITION_TEXT = {
    "turnkey_updated": [
        "Recently renovated throughout with a remodeled kitchen, quartz counters, and updated baths",
        "Move-in ready: new flooring, fresh paint, and a fully updated kitchen",
        "Beautifully upgraded - renovated kitchen and bathrooms, modern fixtures, nothing left to do",
        "Turnkey condition with recent remodel including new appliances and luxury vinyl floors",
    ],
    "original_condition": [
        "Original condition and lovingly maintained - a blank canvas for your personal touches",
        "Sold in its original condition; bring your vision and build sweat equity",
        "Well kept but largely original throughout, priced accordingly",
        "Original kitchen and baths - an opportunity to customize to your taste",
    ],
    "fixer_upper": [
        "Investor special: property needs significant work and is priced for a contractor or cash buyer",
        "Handyman special - needs TLC throughout, sold strictly as-is",
        "Fixer upper with great bones; renovation needed before move-in",
        "Attention investors: full rehab opportunity, do not miss the upside",
    ],
}
TENANT_TEXT = [
    "Currently tenant occupied on a lease through {lease_end} - do not disturb occupants",
    "Property is leased; tenant in place through {lease_end}, showings with 48-hour notice only",
    "Tenant-occupied and sold subject to the existing lease ending {lease_end}",
]
HOA_TEXT = [
    "HOA dues are ${hoa}/month covering exterior maintenance and common areas",
    "Monthly association dues of ${hoa} include water and community amenities",
    "Association fee: ${hoa} per month",
]
FLOOD_TEXT = [
    "Property is located in a designated flood zone - flood insurance will be required by lenders",
    "Seller reports the home sits in flood plain AE; buyers to verify insurance requirements",
    "Flood zone property - flood insurance required",
]
PRICE_CUT_TEXT = [
    "Price reduced - motivated seller, bring all offers",
    "New price! Seller is motivated and open to offers",
    "Priced to sell below recent comparable sales",
]
POOL_TEXT = [
    "Backyard features a sparkling pool and spa, perfect for summer",
    "Enjoy the private pool, patio, and mature landscaping",
    "Resort-style backyard with pool and built-in BBQ",
]
SOLAR_TEXT_OWNED = [
    "Owned solar panels keep electricity bills low",
    "Solar system is fully owned and conveys with the sale",
]
SOLAR_TEXT_LEASED = [
    "Solar panels are leased - buyer to assume the existing solar lease",
    "Leased solar agreement in place; details available on request",
]
ADU_TEXT = [
    "Detached ADU in the back offers rental income or guest quarters",
    "Bonus: a permitted accessory dwelling unit for extended family or rental income",
]
DISTRACTORS = [
    "Close to schools, shopping, and freeway access",
    "Open house this Saturday from 12 to 3",
    "Bright and airy with plenty of natural light throughout",
    "Community amenities include parks and walking trails nearby",
    "Easy access to dining, entertainment, and major employers",
    "A rare opportunity in a quiet, established neighborhood",
]
LEASE_ENDS = ["March 2027", "June 2027", "August 2027", "December 2026"]


def subtype_phrase(subtype: str, rng: np.random.Generator) -> str:
    opts = SUBTYPE_PHRASE.get(subtype, ["home"])
    return str(rng.choice(opts))


def main() -> None:
    df = pd.read_csv(SRC)
    rng = np.random.default_rng(SEED)
    n = len(df)
    year_built = df["YearBuilt"].fillna(1990).astype(int)
    age = 2026 - year_built

    # --- draw mention labels -------------------------------------------------
    # Condition correlates with age so the classifier task is learnable but
    # not deterministic, and the hedonic ablation has a real channel to test.
    p_fixer = np.clip(0.04 + 0.004 * np.maximum(age - 30, 0), 0.04, 0.30)
    p_turnkey = np.clip(0.55 - 0.004 * np.maximum(age - 30, 0), 0.25, 0.55)
    u = rng.random(n)
    condition = np.where(u < p_fixer, "fixer_upper",
                         np.where(u < p_fixer + p_turnkey, "turnkey_updated",
                                  "original_condition"))
    is_condo_like = df["PropertySubType"].isin(["Condominium", "Townhouse"]).to_numpy()
    tenant = rng.random(n) < 0.12
    hoa_present = rng.random(n) < np.where(is_condo_like, 0.85, 0.18)
    hoa_amount = np.where(hoa_present, rng.integers(85, 721, n), 0)
    flood = rng.random(n) < 0.07
    price_cut = rng.random(n) < np.clip(0.10 + 0.002 * df["DaysOnMarket"].fillna(30), 0.10, 0.45)
    pool = rng.random(n) < 0.22
    solar_draw = rng.random(n)
    solar = np.where(solar_draw < 0.06, "owned", np.where(solar_draw < 0.10, "leased", "none"))
    adu = rng.random(n) < 0.06

    remarks, labels = [], []
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        s = []
        s.append(rng.choice(OPENERS).format(
            subtype_phrase=subtype_phrase(row["PropertySubType"], rng), city=row["City"]))
        s.append(rng.choice(LAYOUT).format(
            beds=int(row["BedroomsTotal"]), baths=int(row["BathroomsTotalInteger"]),
            sqft=f"{int(row['LivingArea']):,}"))
        s.append(rng.choice(YEAR_LOT).format(year=int(row["YearBuilt"]),
                                            lot=round(float(row["LotSizeAcres"]), 2)))
        s.append(rng.choice(CONDITION_TEXT[condition[idx]]))
        if tenant[idx]:
            s.append(rng.choice(TENANT_TEXT).format(lease_end=rng.choice(LEASE_ENDS)))
        if hoa_present[idx]:
            s.append(rng.choice(HOA_TEXT).format(hoa=int(hoa_amount[idx])))
        if flood[idx]:
            s.append(rng.choice(FLOOD_TEXT))
        if price_cut[idx]:
            s.append(rng.choice(PRICE_CUT_TEXT))
        if pool[idx]:
            s.append(rng.choice(POOL_TEXT))
        if solar[idx] == "owned":
            s.append(rng.choice(SOLAR_TEXT_OWNED))
        elif solar[idx] == "leased":
            s.append(rng.choice(SOLAR_TEXT_LEASED))
        if adu[idx]:
            s.append(rng.choice(ADU_TEXT))
        for _ in range(rng.integers(1, 3)):
            s.append(rng.choice(DISTRACTORS))
        rng.shuffle(s[1:])  # keep the opener first, shuffle the rest
        remarks.append(". ".join(x.rstrip(".") for x in s) + ".")
        labels.append({
            "ListingId": row["ListingId"],
            "condition": condition[idx],
            "tenant_occupied": int(tenant[idx]),
            "hoa_amount": int(hoa_amount[idx]),
            "flood_mention": int(flood[idx]),
            "price_reduced_mention": int(price_cut[idx]),
            "pool_mention": int(pool[idx]),
            "solar_mention": solar[idx],
            "adu_mention": int(adu[idx]),
        })

    out_dir = HERE / "data"
    out_dir.mkdir(exist_ok=True)
    text_df = pd.DataFrame({"ListingId": df["ListingId"], "PublicRemarks": remarks})
    gold = pd.DataFrame(labels)
    text_df.to_csv(out_dir / "property_text.csv", index=False)
    gold.to_csv(out_dir / "gold_labels.csv", index=False)
    print(f"wrote {len(text_df)} remarks; condition mix: "
          f"{gold['condition'].value_counts().to_dict()}")
    print(f"tenant {gold['tenant_occupied'].mean():.3f} | hoa>0 "
          f"{(gold['hoa_amount'] > 0).mean():.3f} | flood {gold['flood_mention'].mean():.3f}")


if __name__ == "__main__":
    main()
