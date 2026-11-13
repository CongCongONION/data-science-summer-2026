"""
hedonic_model.py — hedonic pricing model with out-of-sample validation

Author : Anna

The base project stops at features and segments. This module adds the
modelling layer a reviewer actually asks about: *how well do property
characteristics explain price, and does that relationship hold on data the
model has never seen?*

Specification (log-log hedonic, OLS estimated with numpy — no black box):
  log(ClosePrice) ~ log(LivingArea) + Bedrooms + Bathrooms
                    + log1p(LotSizeAcres) + PropertyAge + DaysOnMarket
                    + rate_30yr_fixed + County effects + SubType effects

Validation design (the part that makes it a validation exercise, not a fit):
  * TIME-BASED split — train on closes before the split date, test on
    closes after it. A random split would leak market conditions across
    the boundary and flatter the metrics.
  * BENCHMARK — a naive median $/sqft predictor (train median ppsf x area)
    is scored on the same test rows; the model must beat it to claim value.
  * Dollar metrics use the standard smearing correction exp(sigma^2 / 2)
    when transforming log predictions back to prices.
  * Coefficients ship with standard errors and t-statistics, so every
    claimed driver carries its own evidence.

Deliberate exclusions: price_ratio / close_to_orig_list_ratio are NOT used
as predictors — they are built from ClosePrice itself (target leakage).

Outputs (output/): hedonic_coefficients.csv, model_facts.json

  python hedonic_model.py [--split 2025-07-01]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
REFERENCE_YEAR = 2026


def _read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def design(df: pd.DataFrame):
    """Build the feature frame; returns (frame, predictor names, X)."""
    d = df.copy()
    d["CloseDate"] = pd.to_datetime(d["CloseDate"], errors="coerce")
    d["log_price"] = np.log(d["ClosePrice"])
    d["log_area"] = np.log(d["LivingArea"])
    d["log_lot"] = np.log1p(d["LotSizeAcres"].fillna(0).clip(lower=0))
    d["age"] = (REFERENCE_YEAR - d["YearBuilt"]).clip(0, 130)
    base = ["log_area", "BedroomsTotal", "BathroomsTotalInteger", "log_lot",
            "age", "DaysOnMarket", "rate_30yr_fixed"]
    counties = pd.get_dummies(d["CountyOrParish"], prefix="county",
                              drop_first=True, dtype=float)
    subtypes = pd.get_dummies(d["PropertySubType"], prefix="subtype",
                              drop_first=True, dtype=float)
    X = pd.concat([d[base], counties, subtypes], axis=1)
    keep = d["CloseDate"].notna() & d["log_price"].notna() \
        & X.notna().all(axis=1)
    return d.loc[keep], list(X.columns), X.loc[keep]


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS with classical standard errors, via least squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot
    return {"beta": beta, "se": se, "sigma2": sigma2, "r2": r2,
            "resid": resid, "dof": dof}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    return {
        "r2": round(1.0 - float(err @ err)
                    / float(((y_true - y_true.mean()) ** 2).sum()), 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
        "mae": round(float(np.abs(err).mean()), 2),
        "mape_pct": round(float((np.abs(err) / y_true).mean() * 100), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2025-07-01",
                    help="train = closes before this date, test = on/after")
    args = ap.parse_args()
    split = pd.Timestamp(args.split)

    df = _read(os.path.join(OUT, "sold_analysis_ready.csv"))
    d, names, X = design(df)
    train = d["CloseDate"] < split
    Xtr = np.column_stack([np.ones(train.sum()), X.loc[train].to_numpy()])
    Xte = np.column_stack([np.ones((~train).sum()), X.loc[~train].to_numpy()])
    ytr, yte = d.loc[train, "log_price"].to_numpy(), \
        d.loc[~train, "log_price"].to_numpy()
    fit = ols(Xtr, ytr)

    cols = ["intercept"] + names
    table = pd.DataFrame({
        "term": cols, "coef": np.round(fit["beta"], 4),
        "std_err": np.round(fit["se"], 4),
        "t_stat": np.round(fit["beta"] / fit["se"], 2)})
    table.to_csv(os.path.join(OUT, "hedonic_coefficients.csv"), index=False)

    smear = float(np.exp(fit["sigma2"] / 2.0))
    pred_log_te = Xte @ fit["beta"]
    pred_price = np.exp(pred_log_te) * smear
    actual_price = np.exp(yte)
    test_metrics = metrics(actual_price, pred_price)
    train_metrics = metrics(np.exp(ytr),
                            np.exp(Xtr @ fit["beta"]) * smear)

    # benchmark: train median $/sqft x living area, same test rows
    bench_ppsf = float(d.loc[train, "ppsf"].median())
    bench_pred = bench_ppsf * d.loc[~train, "LivingArea"].to_numpy()
    bench_metrics = metrics(actual_price, bench_pred)

    resid = fit["resid"]
    facts = {
        "spec": "log(ClosePrice) ~ log(LivingArea) + beds + baths + "
                "log1p(lot acres) + age + DOM + 30yr rate + county FE + "
                "subtype FE (OLS)",
        "split_date": args.split,
        "n_train": int(train.sum()), "n_test": int((~train).sum()),
        "train_r2_log": round(fit["r2"], 4),
        "train_metrics_price": train_metrics,
        "test_metrics_price": test_metrics,
        "benchmark_median_ppsf": bench_ppsf,
        "benchmark_test_metrics_price": bench_metrics,
        "test_rmse_vs_benchmark_pct": round(
            100 * (test_metrics["rmse"] / bench_metrics["rmse"] - 1), 1),
        "log_area_elasticity": float(table.loc[
            table["term"] == "log_area", "coef"].iloc[0]),
        "residual_log_std": round(float(resid.std()), 4),
        "residual_log_skew": round(float(pd.Series(resid).skew()), 3),
        "top_drivers": table.reindex(
            table["t_stat"].abs().sort_values(ascending=False).index)
            .head(5)[["term", "coef", "t_stat"]].to_dict("records"),
    }
    with open(os.path.join(OUT, "model_facts.json"), "w",
              encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2)
    print(f"  hedonic OLS: train {facts['n_train']:,} / test "
          f"{facts['n_test']:,} rows (split {args.split})")
    print(f"  in-sample R2 (log) {facts['train_r2_log']:.3f} | "
          f"out-of-sample price R2 {test_metrics['r2']:.3f}, "
          f"RMSE ${test_metrics['rmse']:,.0f}, MAPE "
          f"{test_metrics['mape_pct']:.1f}%")
    print(f"  benchmark (median $/sqft): test RMSE "
          f"${bench_metrics['rmse']:,.0f} -> model is "
          f"{facts['test_rmse_vs_benchmark_pct']:+.1f}% vs benchmark "
          f"(log-area elasticity {facts['log_area_elasticity']:.2f})")
    print("  wrote output/hedonic_coefficients.csv, output/model_facts.json")


if __name__ == "__main__":
    main()
