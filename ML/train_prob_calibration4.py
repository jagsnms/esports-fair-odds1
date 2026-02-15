#!/usr/bin/env python3
"""
Probability Calibration Trainer (Series-level)  v1.0

Goal:
  Make p_fair behave like a calibrated win-probability:
    'When we say 0.70, it wins ~70% of the time.'

Inputs (expected in project root):
  - inplay_kappa_logs_clean.csv          (snapshots)
  - inplay_match_results_clean.csv       (final outcomes)

Outputs (written to project root):
  - p_calibration.json
  - p_calibration_report.json

Method:
  - Join snapshots to results by (game, match_id)
  - Convert each snapshot into a labeled example: (p_raw, y)
      y = 1 if Team A won the SERIES, else 0
  - Fit a monotone calibration map using isotonic regression via PAVA (implemented here)
  - Report before/after: ECE, Brier, LogLoss, reliability bins

Notes:
  - This calibrates the *center* probability (p_fair). Your kappa/band calibration is separate.
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

SNAPSHOT_CSV = PROJECT_DIR / "inplay_kappa_logs_clean.csv"
RESULTS_CSV  = PROJECT_DIR / "inplay_match_results_clean.csv"

OUT_JSON   = PROJECT_DIR / "p_calibration.json"
OUT_REPORT = PROJECT_DIR / "p_calibration_report.json"

LEVELS = [
    ("Level 1: Barely usable", 30),
    ("Level 2: Somewhat useful", 75),
    ("Level 3: Solid", 150),
    ("Level 4: Reliable", 300),
    ("Level 5: Beast mode", 500),
]

N_BINS = 10

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def progress_bar(current: int, target: int, width: int = 28):
    frac = 0.0 if target <= 0 else max(0.0, min(1.0, current / target))
    filled = int(round(frac * width))
    bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    return bar, frac

def pick_level(n_matches: int):
    for name, target in LEVELS:
        if n_matches < target:
            bar, frac = progress_bar(n_matches, target)
            return name, target, frac, bar
    name, target = LEVELS[-1]
    bar = "[" + ("#" * 28) + "]"
    return f"{name} (MAX)", target, 1.0, bar

def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1.0 - 1e-6)

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")

def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip01(p)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))) if len(p) else float("nan")

def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    if len(p) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        pb = float(np.mean(p[m]))
        yb = float(np.mean(y[m]))
        w = float(np.mean(m))
        e += w * abs(yb - pb)
    return float(e)

def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> List[Dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    out: List[Dict] = []
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            out.append({
                "bin": b, "p_min": float(bins[b]), "p_max": float(bins[b+1]),
                "n": 0, "p_mean": None, "y_rate": None
            })
            continue
        out.append({
            "bin": b, "p_min": float(bins[b]), "p_max": float(bins[b+1]),
            "n": int(np.sum(m)), "p_mean": float(np.mean(p[m])), "y_rate": float(np.mean(y[m]))
        })
    return out

def pava_isotonic(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = w[order].astype(float)

    blocks = []
    for xi, yi, wi in zip(x, y, w):
        blocks.append([xi, xi, yi * wi, wi])  # lo, hi, sum_yw, sum_w
        while len(blocks) >= 2:
            b1 = blocks[-2]
            b2 = blocks[-1]
            m1 = b1[2] / b1[3]
            m2 = b2[2] / b2[3]
            if m1 <= m2:
                break
            merged = [b1[0], b2[1], b1[2] + b2[2], b1[3] + b2[3]]
            blocks = blocks[:-2] + [merged]

    xs = []
    ys = []
    for lo, hi, sum_yw, sum_w in blocks:
        m = sum_yw / sum_w
        xs.append((lo + hi) / 2.0)
        ys.append(m)

    xs = np.clip(np.array(xs, dtype=float), 0.0, 1.0)
    ys = np.clip(np.array(ys, dtype=float), 0.0, 1.0)
    return xs, ys

def piecewise_linear_map(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return xq
    order = np.argsort(xs)
    xs = xs[order]; ys = ys[order]
    xs2 = np.concatenate([[0.0], xs, [1.0]])
    ys2 = np.concatenate([[ys[0]], ys, [ys[-1]]])
    return np.interp(xq, xs2, ys2)

def infer_series_label(rows: pd.DataFrame) -> np.ndarray:
    """Infer y=1 if Team A won the SERIES else 0, using results winner text.

    Important: snapshots can have messy team_a/team_b strings;
    when a merge suffix exists, prefer team_a_res/team_b_res from results.
    """
    win_col = None
    for c in ["winner_side", "winner", "winner_team", "winner_name", "series_winner", "win"]:
        if c in rows.columns:
            win_col = c
            break
    if win_col is None:
        raise ValueError("Results file missing a winner column (winner/winner_side/etc).")

    w = rows[win_col]
    if pd.api.types.is_numeric_dtype(w):
        return (w.astype(float) >= 0.5).astype(int).to_numpy()

    w_str = w.astype(str).str.strip().str.lower()
    # If the results encode winner as A/B
    if w_str.isin(["a","team a","teama","b","team b","teamb"]).all():
        return w_str.isin(["a","team a","teama"]).astype(int).to_numpy()

    # Prefer result-side team columns if present (merge suffix).
    ta_col = "team_a_res" if "team_a_res" in rows.columns else ("team_a" if "team_a" in rows.columns else None)
    tb_col = "team_b_res" if "team_b_res" in rows.columns else ("team_b" if "team_b" in rows.columns else None)
    if ta_col and tb_col:
        ta = rows[ta_col].astype(str).str.strip().str.lower()
        tb = rows[tb_col].astype(str).str.strip().str.lower()
        y = (w_str == ta).astype(float)
        neither = ~((w_str == ta) | (w_str == tb))
        y[neither] = np.nan
        return y.to_numpy()

    raise ValueError("Cannot infer winner label: missing team_a/team_b columns.")
def main():
    if not SNAPSHOT_CSV.exists():
        raise FileNotFoundError(f"Missing snapshot CSV: {SNAPSHOT_CSV}")
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results CSV: {RESULTS_CSV} (required for outcome calibration)")

    snap = pd.read_csv(SNAPSHOT_CSV)
    res  = pd.read_csv(RESULTS_CSV)

    for df in (snap, res):
        if "game" not in df.columns or "match_id" not in df.columns:
            raise ValueError("Both CSVs must contain 'game' and 'match_id'.")
        df["game"] = df["game"].astype(str).str.lower().str.strip()
        df["match_id"] = df["match_id"].astype(str).str.strip()

    if "p_fair" not in snap.columns:
        raise ValueError("Snapshot CSV missing 'p_fair'.")

    snap["p_fair"] = snap["p_fair"].astype(float)

    merged = snap.merge(res, on=["game","match_id"], how="inner", suffixes=("", "_res"))
    if len(merged) == 0:
        raise ValueError("No joined rows: check that match_id/game match across files.")

    merged = merged.copy()
    merged["y"] = infer_series_label(merged)
    # Report how many rows could not be labeled (winner name didn't match either team).
    if merged["y"].isna().any():
        bad = merged[merged["y"].isna()].copy()
        by_game = bad.groupby("game").size().to_dict()
        print(f"[WARN] Dropping unlabeled rows where winner doesn't match Team A/B: {by_game}")
    merged = merged.dropna(subset=["y"])
    merged["y"] = merged["y"].astype(int)

    report = {
        "project_dir": str(PROJECT_DIR),
        "snapshots_csv": str(SNAPSHOT_CSV),
        "results_csv": str(RESULTS_CSV),
        "ran_at": _now_str(),
        "notes": [
            "Calibrates p_fair (series win prob) vs final series outcomes.",
            "Method: isotonic regression (PAVA) monotone mapping.",
            f"Reliability bins = {N_BINS}",
        ],
        "games": {},
    }

    out_json: Dict[str, Dict] = {}

    print("\n" + "=" * 72)
    print(" PROBABILITY CALIBRATION  v1.0  (p_fair -> calibrated win prob)")
    print("=" * 72)
    print(f"Joined rows: {len(merged)}")
    print("-" * 72)

    for game in sorted(merged["game"].unique().tolist()):
        g = merged[merged["game"] == game].copy()
        n_matches = int(g["match_id"].nunique(dropna=False))
        level_name, target, frac, bar = pick_level(n_matches)

        p_raw = _clip01(g["p_fair"].to_numpy(dtype=float))
        y = g["y"].to_numpy(dtype=int)

        xs, ys = pava_isotonic(p_raw, y)
        p_cal = piecewise_linear_map(xs, ys, p_raw)

        mb = {"ece": ece(p_raw, y), "brier": brier(p_raw, y), "logloss": logloss(p_raw, y)}
        ma = {"ece": ece(p_cal, y), "brier": brier(p_cal, y), "logloss": logloss(p_cal, y)}

        out_json[game] = {
            "method": "isotonic_pava_piecewise_linear",
            "knots": [[float(x), float(v)] for x, v in zip(xs.tolist(), ys.tolist())],
            "trained_on": {"n_rows": int(len(g)), "n_matches": n_matches},
            "metrics_before": mb,
            "metrics_after": ma,
        }

        report["games"][game] = {
            "rows": int(len(g)),
            "matches": n_matches,
            "level": {"name": level_name, "target": target, "progress": float(frac)},
            "metrics_before": mb,
            "metrics_after": ma,
            "reliability_before": reliability_table(p_raw, y),
            "reliability_after": reliability_table(p_cal, y),
        }

        print(f"\n[{game.upper()}] matches={n_matches} rows={len(g)}")
        print(f"  {level_name} -> target {target}")
        print(f"  XP {bar}  {frac*100:5.1f}% ({n_matches}/{target})")
        print(f"  Before: ECE={mb['ece']:.4f}  Brier={mb['brier']:.4f}  LogLoss={mb['logloss']:.4f}")
        print(f"  After : ECE={ma['ece']:.4f}  Brier={ma['brier']:.4f}  LogLoss={ma['logloss']:.4f}")
        print(f"  Knots: {len(out_json[game]['knots'])}")

    OUT_JSON.write_text(json.dumps(out_json, indent=2, sort_keys=True), encoding="utf-8")
    OUT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "-" * 72)
    print(f"Saved: {OUT_JSON.name}")
    print(f"Saved: {OUT_REPORT.name}")
    print("-" * 72)

if __name__ == "__main__":
    main()