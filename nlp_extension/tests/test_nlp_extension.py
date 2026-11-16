"""Integrity checks for the NLP extension artifacts.

These tests verify that the committed data and metrics are internally
consistent: the generated text covers exactly the analysis-ready records,
the gold labels respect their definitions, and every published metric is
a real number in range (re-derived bounds, not re-run models).
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def _load(name):
    return pd.read_csv(HERE / "data" / name)


def test_text_covers_analysis_ready_records():
    src = pd.read_csv(HERE.parent / "reproduction" / "output" / "sold_analysis_ready.csv",
                      usecols=["ListingId"])
    text = _load("property_text.csv")
    gold = _load("gold_labels.csv")
    assert len(text) == len(src) == 5641
    assert set(text["ListingId"]) == set(src["ListingId"])
    assert set(gold["ListingId"]) == set(src["ListingId"])
    assert text["PublicRemarks"].str.len().min() > 100


def test_gold_label_definitions():
    gold = _load("gold_labels.csv")
    assert set(gold["condition"]) == {"turnkey_updated", "original_condition", "fixer_upper"}
    assert set(gold["solar_mention"]) <= {"owned", "leased", "none"}
    for col in ["tenant_occupied", "flood_mention", "price_reduced_mention",
                "pool_mention", "adu_mention"]:
        assert set(gold[col]) <= {0, 1}
    assert (gold["hoa_amount"] >= 0).all()


def test_metrics_files_in_range():
    out = HERE / "output"
    ext = json.loads((out / "extraction_metrics.json").read_text())
    assert ext["n"] == 5641
    for field in ["tenant_occupied", "flood_mention", "price_reduced_mention",
                  "pool_mention", "adu_mention"]:
        assert 0.0 <= ext[field]["f1"] <= 1.0
    clf = json.loads((out / "classification_metrics.json").read_text())
    assert clf["n_train"] + clf["n_test"] == 5641
    assert 0.0 <= clf["macro_f1"] <= 1.0
    ret = json.loads((out / "retrieval_metrics.json").read_text())
    for system in ["tfidf", "minilm_faiss"]:
        for m in ["recall@5", "recall@10", "mrr@10", "ndcg@10"]:
            assert 0.0 <= ret["systems"][system][m] <= 1.0
    abl = json.loads((out / "ablation_metrics.json").read_text())
    assert abl["baseline"]["n_train"] == 3271 and abl["baseline"]["n_test"] == 2370
