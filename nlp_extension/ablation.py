"""Ablation: do text-derived features improve the hedonic pricing benchmark?

Same protocol as reproduction/hedonic_model.py: log-log OLS, time split at
2025-07-01 (train 3,271 / test 2,370), metrics on the price scale.

Three specifications:
  * baseline      - the published hedonic spec, nothing from text
  * + gold text   - adds the TRUE mention labels (upper bound: perfect NLP)
  * + extracted   - adds the labels produced by entity_extraction.py
                    (the deployable pipeline, extraction noise included)

The gap between the gold and extracted rows is the cost of extraction
error; the gap between baseline and gold is the most the text channel
could ever add under this design. Because the synthetic condition label
was drawn conditional on property age (which is already a regressor),
a small or zero gain is the expected, honest outcome - and reporting it
is the point of an ablation.

Run:  python ablation.py  ->  output/ablation_metrics.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from entity_extraction import extract

HERE = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2025-07-01")


def design(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({
        "log_area": np.log(df["LivingArea"]),
        "beds": df["BedroomsTotal"],
        "baths": df["BathroomsTotalInteger"],
        "log_lot": np.log1p(df["LotSizeAcres"]),
        "age": 2026 - df["YearBuilt"],
        "dom": df["DaysOnMarket"],
        "rate": df["rate_30yr_fixed"],
    })
    X = pd.concat([X, pd.get_dummies(df["CountyOrParish"], prefix="county", drop_first=True),
                   pd.get_dummies(df["PropertySubType"], prefix="subtype", drop_first=True),
                   df[extra]], axis=1)
    return X.astype(float)


def run(df: pd.DataFrame, extra: list[str]) -> dict:
    tr = df[df["CloseDate"] < SPLIT]
    te = df[df["CloseDate"] >= SPLIT]
    model = LinearRegression().fit(design(tr, extra), np.log(tr["ClosePrice"]))
    pred = np.exp(model.predict(design(te, extra)))
    return {"n_train": int(len(tr)), "n_test": int(len(te)),
            "r2_price": round(float(r2_score(te["ClosePrice"], pred)), 4),
            "rmse_price": round(float(mean_squared_error(te["ClosePrice"], pred) ** 0.5), 0)}


def main() -> None:
    src = pd.read_csv(HERE.parent / "reproduction" / "output" / "sold_analysis_ready.csv")
    src["CloseDate"] = pd.to_datetime(src["CloseDate"])
    text = pd.read_csv(HERE / "data" / "property_text.csv")
    gold = pd.read_csv(HERE / "data" / "gold_labels.csv")
    df = src.merge(gold, on="ListingId").merge(text, on="ListingId")
    ext = df["PublicRemarks"].apply(extract).apply(pd.Series)
    for col in ["tenant_occupied", "flood_mention", "pool_mention", "price_reduced_mention"]:
        df[f"{col}_ext"] = ext[col].astype(float)
    df["hoa_amount_ext"] = ext["hoa_amount"].astype(float)
    df["condition_pred"] = ext["condition"]

    gold_feats = pd.get_dummies(df["condition"], prefix="cond", drop_first=True).astype(float)
    ext_feats = pd.get_dummies(df["condition_pred"], prefix="cond", drop_first=True).astype(float)
    df = pd.concat([df, gold_feats.add_suffix("_gold"), ext_feats.add_suffix("_ext")], axis=1)
    gold_cols = [c for c in df.columns if c.endswith("_gold")] + \
                ["tenant_occupied", "flood_mention", "pool_mention", "hoa_amount"]
    ext_cols = [c for c in df.columns if c.endswith("_ext")]

    results = {"baseline": run(df, []),
               "plus_gold_text": run(df, gold_cols),
               "plus_extracted_text": run(df, ext_cols)}
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    with open(out / "ablation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
