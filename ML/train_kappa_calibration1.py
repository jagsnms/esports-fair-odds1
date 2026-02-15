
"""Train in-play kappa calibration multipliers from clean logs.

Usage:
  python -m ml.train_kappa_calibration
or:
  python ml/train_kappa_calibration.py

This script reads the clean log CSV produced by app35_ml.py and produces
kappa_calibration.json next to the app.

Calibration target: market mid should fall inside your printed beta interval
at approximately the selected band level (80/90/95%), with a small penalty
against overly-wide bands.

It does NOT require a neural net. It's robust, transparent, and fast.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

APP_DIR = Path(__file__).resolve().parent.parent  # /mnt/data
LOG_CSV = APP_DIR / "inplay_kappa_logs_clean.csv"
OUT_JSON = APP_DIR / "kappa_calibration.json"

def phase_bucket(rounds_a: int, rounds_b: int) -> str:
    rp = int(rounds_a) + int(rounds_b)
    if rp >= 24:
        return "ot"
    if rp <= 8:
        return "early"
    if rp <= 18:
        return "mid"
    return "late"

def beta_interval(p: np.ndarray, kappa: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(p.astype(float), 1e-4, 1-1e-4)
    k = np.maximum(2.0, kappa.astype(float))
    a = p * k
    b = (1.0 - p) * k
    tail = (1.0 - float(level)) / 2.0
    lo = beta_dist.ppf(tail, a, b)
    hi = beta_dist.ppf(1.0 - tail, a, b)
    lo = np.clip(lo, 0.01, 0.99)
    hi = np.clip(hi, 0.01, 0.99)
    return lo, hi

def train(
    log_csv: Path = LOG_CSV,
    out_json: Path = OUT_JSON,
    grid_min: float = 0.50,
    grid_max: float = 2.50,
    grid_step: float = 0.05,
    width_penalty: float = 0.05,
    min_rows_per_bucket: int = 200,
) -> dict:
    if not log_csv.exists():
        raise SystemExit(f"Log not found: {log_csv}")

    df = pd.read_csv(log_csv)

    need = {"game","band_level","p_fair","kappa_map","mid","rounds_a","rounds_b"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in log: {missing}")

    for c in ["band_level","p_fair","kappa_map","mid","rounds_a","rounds_b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["band_level","p_fair","kappa_map","mid","rounds_a","rounds_b"])
    df = df[(df["p_fair"] > 0.0) & (df["p_fair"] < 1.0)]
    df = df[(df["mid"] >= 0.0) & (df["mid"] <= 1.0)]
    df = df[(df["band_level"] >= 0.5) & (df["band_level"] <= 0.99)]
    df = df[(df["kappa_map"] >= 2.0)]

    if df.empty:
        raise SystemExit("No usable rows after cleaning.")

    df["game"] = df["game"].astype(str).str.lower()
    df["bucket"] = [phase_bucket(a, b) for a, b in zip(df["rounds_a"], df["rounds_b"])]

    grid = np.round(np.arange(grid_min, grid_max + 1e-9, grid_step), 4)

    out: dict = {}
    for game in sorted(df["game"].unique()):
        dfg = df[df["game"] == game]
        out.setdefault(game, {})
        for lvl in sorted(dfg["band_level"].unique()):
            lvl = float(lvl)
            lvl_key = str(float(lvl))
            out[game].setdefault(lvl_key, {})
            dfgl = dfg[np.isclose(dfg["band_level"], lvl)]
            for bucket in ["early","mid","late","ot"]:
                dfb = dfgl[dfgl["bucket"] == bucket]
                if len(dfb) < min_rows_per_bucket:
                    continue
                p = dfb["p_fair"].to_numpy(float)
                k = dfb["kappa_map"].to_numpy(float)
                mid = dfb["mid"].to_numpy(float)

                best_m = 1.0
                best_loss = float("inf")
                for m in grid:
                    lo, hi = beta_interval(p, k * m, lvl)
                    cover = float(np.mean((mid >= lo) & (mid <= hi)))
                    width = float(np.mean(hi - lo))
                    loss = (cover - lvl) ** 2 + width_penalty * width
                    if loss < best_loss:
                        best_loss = loss
                        best_m = float(m)
                out[game][lvl_key][bucket] = float(best_m)

    out_json.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out

if __name__ == "__main__":
    cal = train()
    print(f"Saved: {OUT_JSON}")
    for g, levels in cal.items():
        print(f"- {g}: {len(levels)} band-level groups")
