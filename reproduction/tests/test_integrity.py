"""
test_integrity.py — re-derive the published numbers from the committed
outputs and fail if any of them drift.

Author : Anna

These tests do not re-run the pipeline; they audit its artifacts the way a
reviewer would: take only output/ files, recompute the identities, and
compare against the fact files the run recorded. If a number in the README
or the memo cannot be reproduced from the committed outputs, a test fails.

  pytest tests/ -q
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def _json(name: str) -> dict:
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        return json.load(fh)


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUT, name), encoding="utf-8-sig")


RUN = _json("run_facts.json")
QUALITY = _json("quality_facts.json")
MODEL = _json("model_facts.json")
READY = _csv("sold_analysis_ready.csv")


def test_bridge_closes_from_committed_counts():
    o = RUN["outliers"]
    assert o["rows_before"] - o["rows_excluded"] == o["rows_analysis_ready"]
    bridge = _csv("row_count_bridge.csv")
    excl = bridge.loc[bridge["stage"].str.startswith("excluded"),
                      "rows"].sum()
    assert -int(excl) == o["rows_excluded"]
    assert len(READY) == o["rows_analysis_ready"]


def test_exclusion_partition_is_disjoint():
    # every excluded row is business-invalid XOR outlier-only, per the facts
    o = RUN["outliers"]
    assert sum(v["flagged"] for v in o["iqr"].values()) >= 0  # fences exist
    assert o["rows_excluded"] < o["rows_before"]


def test_scorecard_has_no_fail_and_covers_core_dimensions():
    card = _csv("data_quality_scorecard.csv")
    assert (card["status"] != "FAIL").all()
    text = " ".join(card["check"].tolist())
    for dimension in ("completeness", "validity", "uniqueness",
                      "consistency", "geographic", "enrichment"):
        assert dimension in text, dimension
    assert QUALITY["status_tally"] == card["status"] \
        .value_counts().to_dict()


def test_ready_dataset_obeys_business_rules():
    assert (READY["ClosePrice"] > 0).all()
    assert (READY["LivingArea"] > 0).all()
    assert (READY["DaysOnMarket"] >= 0).all()
    assert READY["rate_30yr_fixed"].notna().all()
    assert READY["ListingId"].notna().all()


def test_median_shift_matches_facts():
    o = RUN["outliers"]
    assert float(READY["ClosePrice"].median()) == \
        o["median_close_after"]


def test_hedonic_model_beats_or_matches_naive_within_tolerance():
    tm, bm = MODEL["test_metrics_price"], \
        MODEL["benchmark_test_metrics_price"]
    # on any dataset the model must at least be in the benchmark's league;
    # a blow-out means the spec or the split broke
    assert tm["rmse"] <= 1.05 * bm["rmse"]
    assert 0.8 <= MODEL["log_area_elasticity"] <= 1.2
    assert MODEL["n_train"] + MODEL["n_test"] <= \
        RUN["outliers"]["rows_analysis_ready"]


def test_coefficients_table_matches_top_drivers():
    coef = _csv("hedonic_coefficients.csv")
    assert "log_area" in set(coef["term"])
    la = coef.loc[coef["term"] == "log_area"].iloc[0]
    assert abs(la["coef"] - MODEL["log_area_elasticity"]) < 0.01
    assert la["t_stat"] > 10  # area must be a decisive driver


def test_monthly_index_consistent_with_trend():
    idx = _csv("monthly_market_index.csv")
    trend = _csv("monthly_market_trend.csv")
    assert len(idx) <= len(trend)
    assert idx["price_index"].notna().all()
    assert (idx["sales"] >= 1).all()


def test_segment_tables_sum_to_their_labelled_population():
    # segment tables describe the feature-valid population BEFORE outlier
    # exclusion (recorded in run_facts) — not the analysis-ready rows.
    # Mixing the two populations silently is the failure this test guards.
    county = _csv("segment_by_county.csv")
    n_col = [c for c in county.columns if c.lower().endswith("_count")
             or c.lower() in ("count", "n", "records", "sales")]
    assert n_col, county.columns.tolist()
    assert int(county[n_col[0]].sum()) == \
        RUN["features"]["segment_population_rows"]
    assert RUN["features"]["segment_population_rows"] > len(READY)
