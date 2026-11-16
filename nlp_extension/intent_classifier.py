"""Condition classification from remark text: TF-IDF + logistic regression.

Task: predict the property-condition class (turnkey_updated /
original_condition / fixer_upper) from PublicRemarks alone. This is the
"intent classification" component of the extension - the label a risk or
valuation workflow would actually consume (e.g., flag likely fixer-uppers
for collateral review).

Protocol: stratified 80/20 split (seed 42), TF-IDF word 1-2 grams,
logistic regression with balanced class weights. Reported: accuracy,
macro-F1, per-class precision/recall/F1, confusion matrix.

Caveat (stated in every report): the text is template-generated from the
labels, so scores are optimistic versus real MLS remarks. The honest
reading is "the pipeline works end to end and the evaluation harness is
real", not "this F1 transfers to production text".

Run:  python intent_classifier.py  ->  output/classification_metrics.json
                                    ->  output/confusion_matrix.csv
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
LABELS = ["turnkey_updated", "original_condition", "fixer_upper"]


def main() -> None:
    text = pd.read_csv(HERE / "data" / "property_text.csv")
    gold = pd.read_csv(HERE / "data" / "gold_labels.csv")
    df = text.merge(gold, on="ListingId")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["PublicRemarks"], df["condition"], test_size=0.20,
        random_state=42, stratify=df["condition"])
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    report = classification_report(y_te, pred, labels=LABELS, output_dict=True,
                                   zero_division=0)
    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "per_class": {lab: {"precision": round(report[lab]["precision"], 4),
                            "recall": round(report[lab]["recall"], 4),
                            "f1": round(report[lab]["f1-score"], 4),
                            "support": int(report[lab]["support"])}
                      for lab in LABELS},
    }
    cm = confusion_matrix(y_te, pred, labels=LABELS)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                 columns=[f"pred_{l}" for l in LABELS]).to_csv(out / "confusion_matrix.csv")
    with open(out / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # --- template-family holdout -------------------------------------------
    # The random split above is leaky by design: every phrasing template
    # appears in both train and test, so a perfect score only proves the
    # harness works. The honest generalization test holds out entire
    # phrasing families: train on condition-sentence templates 0-1 of each
    # class, test on templates 2-3 (identified by signature phrases).
    signatures = {
        "turnkey_updated": ["recently renovated throughout", "move-in ready:",
                            "beautifully upgraded", "turnkey condition"],
        "original_condition": ["original condition and lovingly maintained",
                               "sold in its original condition",
                               "well kept but largely original",
                               "original kitchen and baths"],
        "fixer_upper": ["investor special:", "handyman special",
                        "fixer upper with great bones", "attention investors:"],
    }

    def template_id(row) -> int:
        t = row["PublicRemarks"].lower()
        for j, sig in enumerate(signatures[row["condition"]]):
            if sig in t:
                return j
        return -1

    df["template_id"] = df.apply(template_id, axis=1)
    assert (df["template_id"] >= 0).all(), "every remark carries one signature"
    tr = df[df["template_id"] <= 1]
    te = df[df["template_id"] >= 2]
    vec2 = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    clf2 = LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000)
    clf2.fit(vec2.fit_transform(tr["PublicRemarks"]), tr["condition"])
    pred2 = clf2.predict(vec2.transform(te["PublicRemarks"]))
    holdout = {
        "protocol": "train on condition-sentence templates 0-1 per class, "
                    "test on held-out templates 2-3",
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "accuracy": round(float(accuracy_score(te["condition"], pred2)), 4),
        "macro_f1": round(float(f1_score(te["condition"], pred2, average="macro")), 4),
    }
    with open(out / "classification_holdout_metrics.json", "w") as f:
        json.dump(holdout, f, indent=2)
    print(json.dumps(holdout, indent=2))


if __name__ == "__main__":
    main()
