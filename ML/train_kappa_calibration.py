#!/usr/bin/env python3
"""
Gamified Kappa Calibration Trainer (v3.1)

- Reads inplay_kappa_logs_clean.csv (or *_clean*.csv if configured)
- Produces:
  - kappa_calibration.json
  - kappa_calibration_report.json
- Prints a "video game" progress panel per game (CS2 / Valorant)
- Always trains a GLOBAL multiplier per (game, band_level) when enough rows exist
- Trains bucket-specific multipliers when each bucket has enough rows
- Objective: market-mid containment inside beta interval at the target band_level

Place this file at: <project>/ml/train_kappa_calibration.py
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# -------------------- Config --------------------

# This script is expected to live in <project>/ml/
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Default log filename (clean logs)
DEFAULT_LOG_CSV = PROJECT_DIR / "inplay_kappa_logs_clean.csv"

# Allow overriding log path via CLI: python train_kappa_calibration.py /path/to/log.csv
def resolve_log_csv(argv: List[str]) -> Path:
    if len(argv) >= 2:
        p = Path(argv[1]).expanduser()
        if not p.is_absolute():
            p = (PROJECT_DIR / p).resolve()
        return p
    # Prefer v2 log if present
    cand = PROJECT_DIR / "inplay_kappa_logs_v2.csv"
    if cand.exists():
        return cand
    return DEFAULT_LOG_CSV


CALIBRATION_JSON = PROJECT_DIR / "kappa_calibration.json"
REPORT_JSON = PROJECT_DIR / "kappa_calibration_report.json"

# Buckets by total rounds (rounds_a + rounds_b)
BUCKETS = [
    ("early", 0, 8),
    ("mid", 9, 18),
    ("late", 19, 23),
    ("ot", 24, 10_000),
]

# "Unlock" thresholds (unique match_id count per game)
LEVELS = [
    ("Level 1: Barely usable", 30),
    ("Level 2: Somewhat useful", 75),
    ("Level 3: Solid", 150),
    ("Level 4: Reliable", 300),
    ("Level 5: Beast mode", 500),
]

# Minimum rows to train:
MIN_ROWS_GLOBAL = 60         # global multiplier needs this many rows per (game, band_level)
MIN_ROWS_PER_BUCKET = 30     # bucket multipliers need this many rows per (game, band_level, bucket)

# Multiplier grid search
MULT_GRID = np.round(np.linspace(0.50, 2.50, 81), 3)  # 0.50..2.50 step ~0.025

# Width penalty: prefer narrower bands when coverage is similar
WIDTH_PENALTY_LAMBDA = 0.02

# -------------------- Helpers --------------------

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def progress_bar(current: int, target: int, width: int = 28) -> Tuple[str, float]:
    if target <= 0:
        return ("[????????????????????????????]", 0.0)
    frac = max(0.0, min(1.0, current / target))
    filled = int(round(frac * width))
    bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
    return bar, frac

def pick_level(n_matches: int) -> Tuple[str, int, float, str]:
    # Find first level not yet reached
    for name, target in LEVELS:
        if n_matches < target:
            bar, frac = progress_bar(n_matches, target)
            return name, target, frac, bar
    # Maxed
    name, target = LEVELS[-1]
    bar = "[" + ("#" * 28) + "]"
    return f"{name} (MAX)", target, 1.0, bar

def bucket_for_total_rounds(total_rounds: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= total_rounds <= hi:
            return name
    return "unknown"

def bucket_for_row(row: pd.Series) -> str:
    """Bucket a snapshot using OT/half flags when available, otherwise total_rounds."""
    # Prefer explicit OT signal if present
    if "is_ot" in row.index:
        try:
            v = row["is_ot"]
            if isinstance(v, str):
                v = v.strip().lower() in ("1","true","yes","y","t")
            if bool(v):
                return "ot"
        except Exception:
            pass
    if "half" in row.index:
        try:
            hv = str(row["half"]).strip().lower()
            if hv in ("ot","overtime"):
                return "ot"
        except Exception:
            pass
    tr = None
    for c in ("total_rounds","round_in_map","snapshot_idx"):
        if c in row.index:
            try:
                tr = int(row[c])
                break
            except Exception:
                tr=None
    if tr is None:
        tr = int(row.get("rounds_a",0)) + int(row.get("rounds_b",0))
    return bucket_for_total_rounds(int(tr))

def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def beta_interval_from_kappa(p: np.ndarray, kappa: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equal-tailed beta interval for probability p with concentration kappa:
      alpha = p*kappa, beta = (1-p)*kappa
    Interval at 'level' (e.g., 0.80) => tails (1-level)/2
    Uses scipy if available; otherwise falls back to a normal approximation (less accurate).
    """
    a = np.clip(p, 1e-6, 1-1e-6) * np.clip(kappa, 1e-3, None)
    b = (1 - np.clip(p, 1e-6, 1-1e-6)) * np.clip(kappa, 1e-3, None)
    tail = (1.0 - level) / 2.0

    try:
        from scipy.stats import beta as sp_beta  # type: ignore
        lo = sp_beta.ppf(tail, a, b)
        hi = sp_beta.ppf(1.0 - tail, a, b)
        return lo.astype(float), hi.astype(float)
    except Exception:
        # Fallback: normal approximation around p with variance p(1-p)/(kappa+1)
        var = (np.clip(p, 1e-6, 1-1e-6) * (1 - np.clip(p, 1e-6, 1-1e-6))) / (np.clip(kappa, 1e-3, None) + 1.0)
        sd = np.sqrt(np.maximum(var, 1e-12))
        # z for two-tailed level
        # approximate inverse CDF for standard normal
        try:
            from scipy.stats import norm  # type: ignore
            z = norm.ppf(1.0 - tail)
        except Exception:
            # rough: 80%≈1.282, 90%≈1.645, 95%≈1.96
            z = {0.8: 1.282, 0.9: 1.645, 0.95: 1.96}.get(round(level, 2), 1.645)
        lo = np.clip(p - z * sd, 0.0, 1.0)
        hi = np.clip(p + z * sd, 0.0, 1.0)
        return lo.astype(float), hi.astype(float)

def coverage_and_width(mid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    inside = (mid >= lo) & (mid <= hi)
    cover = float(np.mean(inside)) if len(mid) else float("nan")
    width = float(np.mean(hi - lo)) if len(mid) else float("nan")
    return cover, width

@dataclass
class TrainResult:
    best_mult: float
    base_cover: float
    best_cover: float
    base_width: float
    best_width: float
    n: int

def train_multiplier(df: pd.DataFrame, band_level: float) -> Optional[TrainResult]:
    """
    Grid search multiplier on kappa_map to match market-mid containment.
    Requires columns: p_fair, kappa_map, mid
    """
    if len(df) == 0:
        return None

    p = df["p_fair"].astype(float).to_numpy()
    k0 = df["kappa_map"].astype(float).to_numpy()
    mid = df["mid"].astype(float).to_numpy()

    # Base
    lo0, hi0 = beta_interval_from_kappa(p, k0, band_level)
    base_cover, base_width = coverage_and_width(mid, lo0, hi0)

    target = float(band_level)
    best_mult = 1.0
    best_score = float("inf")
    best_cover = base_cover
    best_width = base_width

    for m in MULT_GRID:
        lo, hi = beta_interval_from_kappa(p, k0 * m, band_level)
        cover, width = coverage_and_width(mid, lo, hi)
        # score = coverage error + tiny width penalty (prefer narrower if equal coverage error)
        score = abs(cover - target) + WIDTH_PENALTY_LAMBDA * width
        if score < best_score:
            best_score = score
            best_mult = float(m)
            best_cover = float(cover)
            best_width = float(width)

    return TrainResult(
        best_mult=best_mult,
        base_cover=float(base_cover),
        best_cover=float(best_cover),
        base_width=float(base_width),
        best_width=float(best_width),
        n=int(len(df)),
    )

# -------------------- Main --------------------

def main():
    log_csv = resolve_log_csv(sys.argv)
    if not log_csv.exists():
        raise FileNotFoundError(f"Could not find log CSV at {log_csv}")

    df = pd.read_csv(log_csv)

    # Normalize columns
    # Expected columns: game, match_id, band_level, p_fair, kappa_map, mid, rounds_a, rounds_b, ts
    # Required core columns; round counters can come from rounds_a/b OR round_in_map OR snapshot_idx
    required = ["game", "band_level", "p_fair", "kappa_map", "mid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)[:25]}...")
    # Need some way to bucket by progress in map
    if not (('rounds_a' in df.columns and 'rounds_b' in df.columns) or ('round_in_map' in df.columns) or ('snapshot_idx' in df.columns) or ('total_rounds' in df.columns)):
        raise ValueError("Snapshot CSV must include rounds_a/rounds_b, or round_in_map, or snapshot_idx, or total_rounds for bucketing.")

    # Clean / usable
    df["game"] = df["game"].astype(str).str.lower()
    df["band_level"] = df["band_level"].astype(float)
    df["p_fair"] = df["p_fair"].astype(float)
    df["kappa_map"] = df["kappa_map"].astype(float)
    df["mid"] = df["mid"].astype(float)
    if "rounds_a" in df.columns:
        df["rounds_a"] = pd.to_numeric(df["rounds_a"], errors="coerce").fillna(0).astype(int)
    else:
        df["rounds_a"] = 0
    if "rounds_b" in df.columns:
        df["rounds_b"] = pd.to_numeric(df["rounds_b"], errors="coerce").fillna(0).astype(int)
    else:
        df["rounds_b"] = 0

    if "total_rounds" in df.columns:
        df["total_rounds"] = pd.to_numeric(df["total_rounds"], errors="coerce")
    elif "round_in_map" in df.columns:
        df["total_rounds"] = pd.to_numeric(df["round_in_map"], errors="coerce")
    elif "snapshot_idx" in df.columns:
        df["total_rounds"] = pd.to_numeric(df["snapshot_idx"], errors="coerce")
    else:
        df["total_rounds"] = df["rounds_a"] + df["rounds_b"]
    df["total_rounds"] = df["total_rounds"].fillna(0).astype(int)

    # Prefer explicit OT flags when present
    df["bucket"] = df.apply(bucket_for_row, axis=1)

    # mid validity
    usable = df[(df["mid"] >= 0.0) & (df["mid"] <= 1.0)]
    rows_total_raw = int(len(df))
    rows_total_usable = int(len(usable))

    # time range if present
    time_range = None
    if "ts" in df.columns:
        try:
            ts = pd.to_datetime(df["ts"], errors="coerce")
            if ts.notna().any():
                time_range = {"start": str(ts.min()), "end": str(ts.max())}
        except Exception:
            time_range = None

    report: Dict = {
        "project_dir": str(PROJECT_DIR),
        "log_csv": str(log_csv),
        "rows_total_raw": rows_total_raw,
        "rows_total_usable": rows_total_usable,
        "notes": [
            "Coverage target is market-mid containment inside beta interval.",
            f"MIN_ROWS_GLOBAL={MIN_ROWS_GLOBAL}, MIN_ROWS_PER_BUCKET={MIN_ROWS_PER_BUCKET}",
            f"MULT_GRID={float(MULT_GRID.min())}..{float(MULT_GRID.max())} (n={len(MULT_GRID)})",
            "Buckets: early<=8, mid<=18, late<=23, ot>=24 (prefers is_ot/half when available; otherwise total rounds).",
        ],
        "games": {},
        "calibration": {},
        "training": {},
        "ran_at": _now_str(),
    }

    # Gamified print header
    print("\n" + "=" * 72)
    print(" KAPPA CALIBRATION TRAINER  v3.1  (Gamified Progress + Buckets)")
    print("=" * 72)
    print(f"Log: {log_csv}")
    print(f"Rows usable: {rows_total_usable}/{rows_total_raw}")
    if time_range:
        print(f"Time range: {time_range['start']}  ->  {time_range['end']}")
    print("-" * 72)

    calibration_out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for game in sorted(usable["game"].unique().tolist()):
        gdf = usable[usable["game"] == game].copy()

        # Match IDs (may be missing)
        if "match_id" in gdf.columns:
            match_ids = gdf["match_id"].astype(str)
            n_matches = int(match_ids.nunique(dropna=False))
            top_match = match_ids.value_counts(dropna=False).head(5).to_dict()
        else:
            n_matches = 0
            top_match = {}

        # Progress level
        level_name, target, frac, bar = pick_level(n_matches)

        # Buckets count
        bucket_counts = gdf["bucket"].value_counts().to_dict()

        # Band levels present
        band_levels = sorted(gdf["band_level"].unique().tolist())

        # mid stats
        mid_stats = {
            "min": float(gdf["mid"].min()),
            "max": float(gdf["mid"].max()),
            "mean": float(gdf["mid"].mean()),
            "std": float(gdf["mid"].std(ddof=1)) if len(gdf) > 1 else 0.0,
        }

        # Print game panel
        print(f"\n[{game.upper()}]  Matches (unique match_id): {n_matches}")
        print(f"  {level_name}  -> target {target} matches")
        print(f"  XP {bar}  {frac*100:5.1f}% ({n_matches}/{target})")
        print(f"  Rows: {len(gdf)} | Band levels seen: {band_levels}")
        print(f"  Buckets: " + ", ".join([f"{k}:{bucket_counts.get(k,0)}" for k,_,_ in BUCKETS]))
        print(f"  Mid stats: mean={mid_stats['mean']:.3f} std={mid_stats['std']:.3f} range=[{mid_stats['min']:.3f},{mid_stats['max']:.3f}]")

        report["games"][game] = {
            "rows_usable": int(len(gdf)),
            "match_id": {"unique": n_matches, "top": top_match},
            "band_levels": band_levels,
            "buckets": {k: int(bucket_counts.get(k, 0)) for k,_,_ in BUCKETS},
            "mid_stats": mid_stats,
        }

        calibration_out.setdefault(game, {})

        # Train per band_level
        for bl in band_levels:
            bldf = gdf[gdf["band_level"] == bl].copy()
            bl_key = str(round(float(bl), 3))
            calibration_out[game].setdefault(bl_key, {})

            # GLOBAL
            trained_any = False
            if len(bldf) >= MIN_ROWS_GLOBAL:
                res = train_multiplier(bldf, float(bl))
                if res:
                    calibration_out[game][bl_key]["global"] = float(res.best_mult)
                    report["training"].setdefault(game, {}).setdefault(bl_key, {})["global"] = {
                        "n": res.n,
                        "target": float(bl),
                        "base_cover": res.base_cover,
                        "best_cover": res.best_cover,
                        "base_width": res.base_width,
                        "best_width": res.best_width,
                        "best_mult": res.best_mult,
                        "improve_cover_abs": abs(res.best_cover - float(bl)) - abs(res.base_cover - float(bl)),
                    }
                    print(f"  - band {bl:.2f} GLOBAL: n={res.n}  cover {res.base_cover:.3f}->{res.best_cover:.3f}  width {res.base_width:.3f}->{res.best_width:.3f}  mult={res.best_mult:.3f}")
                    trained_any = True
            else:
                print(f"  - band {bl:.2f} GLOBAL: skipped (need {MIN_ROWS_GLOBAL}, have {len(bldf)})")

            # BUCKETS (unlock)
            for bucket, lo, hi in BUCKETS:
                bdf = bldf[bldf["bucket"] == bucket]
                if len(bdf) < MIN_ROWS_PER_BUCKET:
                    continue
                res = train_multiplier(bdf, float(bl))
                if not res:
                    continue
                calibration_out[game][bl_key][bucket] = float(res.best_mult)
                report["training"].setdefault(game, {}).setdefault(bl_key, {})[bucket] = {
                    "n": res.n,
                    "target": float(bl),
                    "base_cover": res.base_cover,
                    "best_cover": res.best_cover,
                    "base_width": res.base_width,
                    "best_width": res.best_width,
                    "best_mult": res.best_mult,
                    "improve_cover_abs": abs(res.best_cover - float(bl)) - abs(res.base_cover - float(bl)),
                }
                if trained_any:
                    print(f"    * {bucket:5s}: n={res.n}  cover {res.base_cover:.3f}->{res.best_cover:.3f}  width {res.base_width:.3f}->{res.best_width:.3f}  mult={res.best_mult:.3f}")
                else:
                    print(f"  - band {bl:.2f} {bucket:5s}: n={res.n}  cover {res.base_cover:.3f}->{res.best_cover:.3f}  width {res.base_width:.3f}->{res.best_width:.3f}  mult={res.best_mult:.3f}")

            # Ensure at least an empty object exists (so your report shows it)
            report["calibration"].setdefault(game, {}).setdefault(bl_key, {})

    # Write outputs
    with open(CALIBRATION_JSON, "w", encoding="utf-8") as f:
        json.dump(calibration_out, f, indent=2, sort_keys=True)

    report["calibration"] = {g: {bl: calibration_out.get(g, {}).get(bl, {}) for bl in report["games"][g]["band_levels"]} for g in report["games"].keys()}

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("\n" + "-" * 72)
    print(f"Saved calibration: {CALIBRATION_JSON.name}")
    print(f"Saved report:      {REPORT_JSON.name}")
    print("-" * 72)
    print("Tip: Buckets unlock automatically as you collect more data. Global trains first.\n")

if __name__ == "__main__":
    main()