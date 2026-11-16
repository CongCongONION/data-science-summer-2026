"""Rule-based entity / signal extraction from the synthetic remarks.

Baseline system: regular expressions for the structured entities (bedrooms,
bathrooms, living area, year built, HOA amount) and keyword rules for the
boolean risk-relevant mentions (tenant occupancy, flood, price reduction,
pool, solar owned/leased, ADU) plus a keyword-scored condition classifier.

Evaluated against ``data/gold_labels.csv``. Because the text was generated
from the same labels (see generate_text_data.py), these scores are an upper
bound on what the same rules would achieve on real MLS remarks; the report
treats them that way. The useful output is the *per-field breakdown*: which
fields the rules recover reliably and where they fail (error analysis in
validation_report.md).

Run:  python entity_extraction.py   ->  output/extraction_metrics.json
                                     ->  output/extraction_by_subtype.csv
"""

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _first_int(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def extract(text: str) -> dict:
    t = text.lower()
    beds = _first_int([r"(\d+)\s*(?:bd\b|bed)", r"(\d+)\s*bedrooms?"], t)
    baths = _first_int([r"(\d+)\s+full\s+baths?", r"(\d+)\s*(?:ba\b|bath)",
                        r"(\d+)\s*bathrooms?"], t)
    sqft = _first_int([r"([\d,]+)\s*(?:sq\s?ft|square feet|sqft|sf)\b"], t)
    year = _first_int([r"built in (\d{4})", r"constructed in (\d{4})", r"built (\d{4})"], t)
    hoa = _first_int([r"hoa dues are \$([\d,]+)", r"association dues of \$([\d,]+)",
                      r"association fee: \$([\d,]+)", r"\$([\d,]+)/month",
                      r"\$([\d,]+) per month"], t)
    tenant = int(any(k in t for k in ["tenant occupied", "tenant-occupied", "tenant in place",
                                      "leased;", "subject to the existing lease"]))
    flood = int("flood" in t)
    price_cut = int(any(k in t for k in ["price reduced", "new price", "priced to sell",
                                         "motivated seller"]))
    pool = int("pool" in t)
    if re.search(r"solar[^.]*leas|leas[^.]*solar", t):
        solar = "leased"
    elif "solar" in t:
        solar = "owned"
    else:
        solar = "none"
    adu = int("adu" in t or "accessory dwelling" in t)
    score = {"turnkey_updated": 0, "original_condition": 0, "fixer_upper": 0}
    for kw in ["renovat", "remodel", "updated", "upgraded", "move-in ready", "turnkey",
               "new flooring", "fresh paint"]:
        if kw in t:
            score["turnkey_updated"] += 1
    for kw in ["original condition", "blank canvas", "sweat equity", "largely original",
               "priced accordingly"]:
        if kw in t:
            score["original_condition"] += 1
    for kw in ["investor special", "handyman", "fixer", "needs tlc", "as-is", "rehab",
               "needs significant work"]:
        if kw in t:
            score["fixer_upper"] += 1
    condition = max(score, key=score.get) if max(score.values()) > 0 else "original_condition"
    return {"beds": beds, "baths": baths, "sqft": sqft, "year_built": year,
            "hoa_amount": hoa or 0, "tenant_occupied": tenant, "flood_mention": flood,
            "price_reduced_mention": price_cut, "pool_mention": pool,
            "solar_mention": solar, "adu_mention": adu, "condition": condition}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main() -> None:
    text = pd.read_csv(HERE / "data" / "property_text.csv")
    gold = pd.read_csv(HERE / "data" / "gold_labels.csv")
    src = pd.read_csv(HERE.parent / "reproduction" / "output" / "sold_analysis_ready.csv",
                      usecols=["ListingId", "BedroomsTotal", "BathroomsTotalInteger",
                               "LivingArea", "YearBuilt", "PropertySubType"])
    df = text.merge(gold, on="ListingId").merge(src, on="ListingId")
    pred = df["PublicRemarks"].apply(extract).apply(pd.Series)
    df = pd.concat([df, pred.add_suffix("_pred")], axis=1)

    metrics = {"n": int(len(df))}
    # numeric entities: exact-match accuracy
    for field, gold_col in [("beds", "BedroomsTotal"), ("baths", "BathroomsTotalInteger"),
                            ("sqft", "LivingArea"), ("year_built", "YearBuilt"),
                            ("hoa_amount", "hoa_amount")]:
        match = (df[f"{field}_pred"].fillna(-1).astype(float) == df[gold_col].astype(float))
        metrics[f"{field}_exact_match"] = round(float(match.mean()), 4)
    # boolean mentions: P/R/F1
    for field in ["tenant_occupied", "flood_mention", "price_reduced_mention",
                  "pool_mention", "adu_mention"]:
        g, p = df[field], df[f"{field}_pred"]
        tp = int(((g == 1) & (p == 1)).sum()); fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        pr, rc, f1 = prf(tp, fp, fn)
        metrics[field] = {"precision": pr, "recall": rc, "f1": f1, "support": int(g.sum())}
    # solar 3-way: accuracy + leased recall (the risk-relevant distinction)
    metrics["solar_accuracy"] = round(float((df["solar_mention"] == df["solar_mention_pred"]).mean()), 4)
    leased = df[df["solar_mention"] == "leased"]
    metrics["solar_leased_recall"] = round(float((leased["solar_mention_pred"] == "leased").mean()), 4)
    # condition: accuracy
    metrics["condition_accuracy"] = round(float((df["condition"] == df["condition_pred"]).mean()), 4)

    # error breakdown by property subtype (condition accuracy + sqft match)
    rows = []
    for subtype, gdf in df.groupby("PropertySubType"):
        rows.append({
            "subtype": subtype, "n": int(len(gdf)),
            "condition_acc": round(float((gdf["condition"] == gdf["condition_pred"]).mean()), 4),
            "sqft_match": round(float((gdf["sqft_pred"].fillna(-1).astype(float) == gdf["LivingArea"].astype(float)).mean()), 4),
            "tenant_f1_recall": round(float(gdf.loc[gdf["tenant_occupied"] == 1, "tenant_occupied_pred"].mean() or 0), 4),
        })
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "extraction_by_subtype.csv", index=False)
    with open(out / "extraction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
