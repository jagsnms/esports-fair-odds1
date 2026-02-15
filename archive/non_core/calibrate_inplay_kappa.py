#!/usr/bin/env python3
"""
Calibrate in-play fair probability (mean) and Kappa (band tightness) from logs.

Inputs (created by the Streamlit app):
- inplay_kappa_logs.csv
- inplay_match_results.csv

Workflow:
1) Make sure every traded match has a stable match_id in the app.
2) During the match, hit "Add snapshot" (and optionally enable "Persist snapshots").
3) After the match ends, use "Settle match" to write winner A/B.
4) Run:  python calibrate_inplay_kappa.py

Outputs:
- Prints mean calibration table + simple Platt fit (a,b)
- Prints K diagnostics and suggests a global kappa_scale
- Writes inplay_calibration.json with suggested parameters
"""
from __future__ import annotations
import json
from pathlib import Path
from math import log, exp
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS = PROJECT_ROOT / "logs" / "inplay_kappa_logs.csv"
RESULTS = PROJECT_ROOT / "logs" / "inplay_match_results_clean.csv"
OUT_JSON = PROJECT_ROOT / "config" / "inplay_calibration.json"

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def platt_fit(p: np.ndarray, y: np.ndarray, iters: int = 4000, lr: float = 0.01) -> tuple[float,float]:
    """
    Fit Platt scaling: p_cal = sigmoid(a + b*logit(p))
    Simple gradient descent (no sklearn dependency).
    """
    x = logit(p)
    a = 0.0
    b = 1.0
    for _ in range(iters):
        z = a + b * x
        p_hat = sigmoid(z)
        # gradients for log loss
        da = np.mean(p_hat - y)
        db = np.mean((p_hat - y) * x)
        a -= lr * da
        b -= lr * db
    return float(a), float(b)

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def main():
    if not LOGS.exists():
        raise SystemExit(f"Missing {LOGS}. Run the app with 'Persist snapshots' enabled.")
    if not RESULTS.exists():
        raise SystemExit(f"Missing {RESULTS}. In the app, use 'Settle match' after matches end.")

    logs = pd.read_csv(LOGS)
    results = pd.read_csv(RESULTS)

    # Deduplicate results by last timestamp per match_id (in case you clicked twice)
    results = results.sort_values("timestamp").groupby(["match_id","game"], as_index=False).tail(1)

    # Winner column is "A" or "B" relative to snapshot's Team A/B.
    merged = logs.merge(results[["match_id","game","winner"]], on=["match_id","game"], how="inner")
    if len(merged) < 200:
        print(f"Warning: only {len(merged)} snapshots with outcomes. Calibration gets much better after ~500+ snapshots.")

    # Outcome as 1 if Team A won, else 0.
    merged["y"] = (merged["winner"].astype(str).str.upper() == "A").astype(int)

    # Use p_fair (already series-adjusted if you selected series scope)
    p = merged["p_fair"].astype(float).clip(1e-6, 1-1e-6).to_numpy()
    y = merged["y"].to_numpy()

    print("\n=== Mean calibration (reliability) ===")
    bins = np.linspace(0, 1, 11)
    merged["bin"] = pd.cut(merged["p_fair"], bins=bins, include_lowest=True)
    rel = merged.groupby("bin").agg(
        n=("y","size"),
        p_mean=("p_fair","mean"),
        win_rate=("y","mean"),
    ).reset_index()
    print(rel.to_string(index=False))
    print(f"\nRaw Brier: {brier(p,y):.4f}")

    # Platt scaling
    a,b = platt_fit(p, y)
    p_cal = sigmoid(a + b*logit(p))
    print(f"\nPlatt scaling fit: a={a:.4f}, b={b:.4f}")
    print(f"Post-Platt Brier: {brier(p_cal,y):.4f}")

    # === K diagnostics ===
    # If your bands are "too wide", you’ll see low surprise but low K (overly conservative).
    # If bands are "too tight", you’ll see lots of big surprises.
    K = merged["kappa_map"].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0).to_numpy()
    abs_err = np.abs(y - p_cal)
    print("\n=== K diagnostics ===")
    print(f"Mean |y - p_cal|: {abs_err.mean():.4f}")
    print(f"Median K: {np.median(K):.2f}  (min {np.min(K):.2f}, max {np.max(K):.2f})")

    # Quick heuristic for a global K scale:
    # target: median abs_err around ~0.20 at midgame; if abs_err small but K tiny -> bands too wide => scale K up
    # This is just a starting point.
    # Compute empirical error by current band width proxy:
    band_w = (merged["band_hi"].astype(float) - merged["band_lo"].astype(float)).clip(0, 1).to_numpy()
    med_w = float(np.median(band_w))
    print(f"Median band width: {med_w:.3f} (probability units)")

    # Suggest global multiplier so median band width ~0.25 at 80% level (tune to taste)
    # Band width ~ 1/sqrt(K), so scaleK ≈ (med_w / target_w)^2 inverse
    target_w = 0.25
    if med_w > 1e-6:
        scaleK = (med_w / target_w) ** 2
        # If bands are wider than target_w, scaleK>1 means you need MORE K; thus multiply K by scaleK
        print(f"Suggested global K multiplier (rough): {scaleK:.2f}")
    else:
        scaleK = 1.0
        print("Could not compute band width; check logs.")

    out = {
        "platt_a": a,
        "platt_b": b,
        "suggested_global_k_multiplier": float(scaleK),
        "n_snapshots": int(len(merged)),
        "median_band_width": med_w,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {OUT_JSON}")

if __name__ == "__main__":
    main()
