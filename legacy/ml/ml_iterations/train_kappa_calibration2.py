"""Train in-play kappa calibration multipliers from clean logs.

Run:
  python train_kappa_calibration.py
or:
  python ml/train_kappa_calibration.py

What it does
------------
Reads your clean in-play snapshot log CSV and learns a *multiplier* on your
logged kappa values so that your beta interval coverage matches the target
band level (e.g., 0.80 / 0.90).

Current target (as implemented):
  Market mid should fall inside your printed beta interval about `band_level`
  of the time, per phase bucket (early/mid/late/ot).

Outputs
-------
- kappa_calibration.json
- kappa_calibration_report.json  (human-readable progress snapshot)

Notes
-----
This is calibration, not a neural net. It's meant to be boring and reliable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


# -----------------------------
# Path discovery / defaults
# -----------------------------

def _find_project_root(start: Path) -> Path:
    """Walk upward looking for a likely project root."""
    start = start.resolve()
    for p in [start, *start.parents]:
        # Prefer a directory that contains your clean log (or any variant).
        if any(p.glob("inplay_kappa_logs_clean*.csv")):
            return p
        if (p / "app35_ml.py").exists() or (p / "app35.py").exists():
            return p
    return start.parent


PROJECT_DIR = _find_project_root(Path(__file__).resolve().parent)

def _pick_latest_clean_log(project_dir: Path) -> Optional[Path]:
    candidates = sorted(project_dir.glob("inplay_kappa_logs_clean*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    # Back-compat: if user uses a generic name without suffix
    generic = project_dir / "inplay_kappa_logs_clean.csv"
    if generic.exists():
        return generic
    return None

LOG_CSV = _pick_latest_clean_log(PROJECT_DIR)
OUT_JSON = PROJECT_DIR / "kappa_calibration.json"
OUT_REPORT = PROJECT_DIR / "kappa_calibration_report.json"


# -----------------------------
# Helpers
# -----------------------------

def phase_bucket(rounds_a: int, rounds_b: int) -> str:
    rp = int(rounds_a) + int(rounds_b)
    if rp >= 24:
        return "ot"
    if rp <= 8:
        return "early"
    if rp <= 18:
        return "mid"
    return "late"

def beta_interval(p: np.ndarray, kappa: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    p = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
    k = np.maximum(2.0, kappa.astype(float))
    a = p * k
    b = (1.0 - p) * k
    tail = (1.0 - float(level)) / 2.0
    lo = beta_dist.ppf(tail, a, b)
    hi = beta_dist.ppf(1.0 - tail, a, b)
    # keep sane bounds
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    return lo, hi

def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _describe_counts(series: pd.Series, top_n: int = 8) -> Dict:
    vc = series.value_counts(dropna=False).head(top_n)
    return {"unique": int(series.nunique(dropna=False)),
            "top": {str(k): int(v) for k, v in vc.items()}}

def _split_train_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by match_id when available to reduce leakage."""
    if "match_id" in df.columns:
        ids = df["match_id"].astype(str)
        uniq = ids.dropna().unique().tolist()
        if len(uniq) >= 5:
            # deterministic split by sorted ids (stable across runs)
            uniq_sorted = sorted(uniq)
            cut = int(np.floor(0.8 * len(uniq_sorted)))
            train_ids = set(uniq_sorted[:cut])
            train = df[ids.isin(train_ids)]
            hold = df[~ids.isin(train_ids)]
            if not hold.empty and not train.empty:
                return train, hold
    # fallback: random split
    df2 = df.sample(frac=1.0, random_state=42)
    cut = int(np.floor(0.8 * len(df2)))
    return df2.iloc[:cut], df2.iloc[cut:]


# -----------------------------
# Training
# -----------------------------

@dataclass
class BucketResult:
    n: int
    base_cover: float
    base_width: float
    best_m: float
    best_cover: float
    best_width: float
    loss: float


def train(
    log_csv: Optional[Path] = LOG_CSV,
    out_json: Path = OUT_JSON,
    out_report: Path = OUT_REPORT,
    grid_min: float = 0.50,
    grid_max: float = 2.50,
    grid_step: float = 0.05,
    width_penalty: float = 0.05,
    min_rows_per_bucket: int = 200,
) -> Dict:
    if log_csv is None or not Path(log_csv).exists():
        raise SystemExit(
            f"No clean log CSV found in {PROJECT_DIR}. Expected a file like inplay_kappa_logs_clean*.csv"
        )

    log_csv = Path(log_csv)
    df = pd.read_csv(log_csv)

    required = {"game", "band_level", "p_fair", "kappa_map", "mid", "rounds_a", "rounds_b"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in log: {missing}")

    _safe_to_numeric(df, ["band_level", "p_fair", "kappa_map", "mid", "rounds_a", "rounds_b",
                          "gap_rounds", "gap_flag", "total_rounds", "prev_total_rounds"])

    # basic cleaning
    df["game"] = df["game"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["band_level", "p_fair", "kappa_map", "mid", "rounds_a", "rounds_b"])
    df = df[(df["p_fair"] > 0.0) & (df["p_fair"] < 1.0)]
    df = df[(df["mid"] >= 0.0) & (df["mid"] <= 1.0)]
    df = df[(df["band_level"] >= 0.5) & (df["band_level"] <= 0.99)]
    df = df[(df["kappa_map"] >= 2.0)]

    if df.empty:
        raise SystemExit("No usable rows after cleaning (check that mid, p_fair, and kappa_map are present).")

    df["bucket"] = [phase_bucket(a, b) for a, b in zip(df["rounds_a"], df["rounds_b"])]

    grid = np.round(np.arange(grid_min, grid_max + 1e-9, grid_step), 4)

    # Summary stats for report
    report: Dict = {
        "project_dir": str(PROJECT_DIR),
        "log_csv": str(log_csv),
        "rows_total_raw": int(len(pd.read_csv(log_csv))),
        "rows_total_usable": int(len(df)),
        "games": {},
        "notes": [
            "Coverage target is market-mid containment inside beta interval.",
            "Holdout split is by match_id when enough unique IDs exist; otherwise random rows.",
            f"min_rows_per_bucket={min_rows_per_bucket}",
        ],
        "training": {},
    }

    for game in sorted(df["game"].unique()):
        dfg = df[df["game"] == game].copy()

        g_summary: Dict = {
            "rows_usable": int(len(dfg)),
            "band_levels": sorted([float(x) for x in dfg["band_level"].unique().tolist()]),
            "buckets": {b: int((dfg["bucket"] == b).sum()) for b in ["early", "mid", "late", "ot"]},
        }
        if "match_id" in dfg.columns:
            g_summary["match_id"] = _describe_counts(dfg["match_id"].astype(str))
        if "map_index" in dfg.columns:
            g_summary["map_index"] = _describe_counts(dfg["map_index"].astype(str))
        if "timestamp" in dfg.columns:
            # timestamps can be messy; keep string min/max
            try:
                ts = pd.to_datetime(dfg["timestamp"], errors="coerce")
                if ts.notna().any():
                    g_summary["time_range"] = {"start": str(ts.min()), "end": str(ts.max())}
            except Exception:
                pass

        # Mid sanity
        g_summary["mid_stats"] = {
            "min": float(dfg["mid"].min()),
            "max": float(dfg["mid"].max()),
            "mean": float(dfg["mid"].mean()),
            "std": float(dfg["mid"].std(ddof=0)),
        }

        report["games"][game] = g_summary

        # Train per band level and bucket
        out: Dict = {}
        train_df, hold_df = _split_train_holdout(dfg)

        for lvl in sorted(train_df["band_level"].unique()):
            lvl = float(lvl)
            lvl_key = f"{lvl:.4g}"
            out.setdefault(lvl_key, {})
            for bucket in ["early", "mid", "late", "ot"]:
                dfb = train_df[(train_df["bucket"] == bucket) & (np.isclose(train_df["band_level"], lvl))]
                if len(dfb) < min_rows_per_bucket:
                    continue

                p = dfb["p_fair"].to_numpy(float)
                k = dfb["kappa_map"].to_numpy(float)
                mid = dfb["mid"].to_numpy(float)

                # base metrics at m=1.0
                base_lo, base_hi = beta_interval(p, k, lvl)
                base_cover = float(np.mean((mid >= base_lo) & (mid <= base_hi)))
                base_width = float(np.mean(base_hi - base_lo))

                best_m = 1.0
                best_loss = float("inf")
                best_cover = base_cover
                best_width = base_width

                for m in grid:
                    lo, hi = beta_interval(p, k * m, lvl)
                    cover = float(np.mean((mid >= lo) & (mid <= hi)))
                    width = float(np.mean(hi - lo))
                    loss = (cover - lvl) ** 2 + width_penalty * width
                    if loss < best_loss:
                        best_loss = loss
                        best_m = float(m)
                        best_cover = cover
                        best_width = width

                out[lvl_key][bucket] = float(best_m)

                # evaluation on holdout for same lvl/bucket
                hold_bucket = hold_df[(hold_df["bucket"] == bucket) & (np.isclose(hold_df["band_level"], lvl))]
                hold_metrics = None
                if len(hold_bucket) >= max(50, int(min_rows_per_bucket * 0.2)):
                    hp = hold_bucket["p_fair"].to_numpy(float)
                    hk = hold_bucket["kappa_map"].to_numpy(float)
                    hmid = hold_bucket["mid"].to_numpy(float)

                    hlo0, hhi0 = beta_interval(hp, hk, lvl)
                    hlo1, hhi1 = beta_interval(hp, hk * best_m, lvl)

                    hold_metrics = {
                        "n": int(len(hold_bucket)),
                        "base_cover": float(np.mean((hmid >= hlo0) & (hmid <= hhi0))),
                        "best_cover": float(np.mean((hmid >= hlo1) & (hmid <= hhi1))),
                        "base_width": float(np.mean(hhi0 - hlo0)),
                        "best_width": float(np.mean(hhi1 - hlo1)),
                    }

                report["training"].setdefault(game, {}).setdefault(lvl_key, {})[bucket] = {
                    "train": {
                        "n": int(len(dfb)),
                        "target": float(lvl),
                        "base_cover": base_cover,
                        "best_cover": best_cover,
                        "base_width": base_width,
                        "best_width": best_width,
                        "best_multiplier": float(best_m),
                        "loss": float(best_loss),
                    },
                    "holdout": hold_metrics,
                }

        # write out per-game calibration into global structure
        report.setdefault("calibration", {})[game] = out

    # Write calibration JSON (the artifact used by the app)
    calibration_out = report.get("calibration", {})
    out_json.write_text(json.dumps(calibration_out, indent=2, sort_keys=True))

    # Compare to previous calibration (if any) for progress visibility
    prev = None
    if out_json.exists():
        try:
            prev = json.loads(out_json.read_text())
        except Exception:
            prev = None

    # Save report
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True))

    # Print a gamified progress snapshot to console
    def _bar(p: float, width: int = 24) -> str:
        p = 0.0 if p is None else float(p)
        p = max(0.0, min(1.0, p))
        filled = int(round(p * width))
        return "[" + ("█" * filled) + ("░" * (width - filled)) + f"] {p*100:5.1f}%"

    LEVELS = [
        ("Barely usable", 30),
        ("Somewhat useful", 75),
        ("Pretty solid", 150),
        ("Strong", 300),
        ("Beast mode", 500),
    ]

    print("\n=== KAPPA CALIBRATION STATUS ===")
    print(f"Log: {log_csv.name}  (usable rows: {len(df)})")
    print(f"Saved calibration: {out_json.name}")
    print(f"Saved report:      {out_report.name}\n")

    for game, gsum in report["games"].items():
        # Progress is based on unique match_id if available; otherwise map_index; otherwise rows as a fallback.
        uniq_matches = None
        if "match_id" in gsum:
            uniq_matches = int(gsum["match_id"].get("unique", 0))
        elif "map_index" in gsum:
            uniq_matches = int(gsum["map_index"].get("unique", 0))
        else:
            uniq_matches = max(1, int(gsum.get("rows_usable", 0) // 40))  # rough fallback

        # Determine level + progress
        prev_cap = 0
        level_name, next_cap = LEVELS[-1]
        for name, cap in LEVELS:
            if uniq_matches < cap:
                level_name, next_cap = name, cap
                break
            prev_cap = cap
        if uniq_matches >= LEVELS[-1][1]:
            prev_cap = LEVELS[-1][1]
            level_name, next_cap = LEVELS[-1][0], LEVELS[-1][1]
            prog = 1.0
        else:
            denom = max(1, (next_cap - prev_cap))
            prog = (uniq_matches - prev_cap) / denom

        mid_stats = gsum["mid_stats"]
        mids = f"mid μ={mid_stats['mean']:.3f} σ={mid_stats['std']:.3f} range[{mid_stats['min']:.3f},{mid_stats['max']:.3f}]"

        print(f"🕹️  {game.upper()}  |  Matches logged: {uniq_matches}")
        print(f"    Level target: {level_name} ({min(uniq_matches, next_cap)}/{next_cap})")
        print(f"    XP bar: {_bar(prog)}")
        print(f"    Rows: {gsum.get('rows_usable', 0)} | band_levels seen: {gsum.get('band_levels', [])} | {mids}")
        bcounts = gsum["buckets"]
        print(f"    Buckets: early={bcounts['early']} mid={bcounts['mid']} late={bcounts['late']} ot={bcounts['ot']}")

        # Training quality summary (coverage error + width changes)
        trained_levels = report["training"].get(game, {})
        if not trained_levels:
            print("    ⚠️  No trained buckets yet (need more usable rows).\n")
            continue
        for lvl_key, buckets in trained_levels.items():
            errs = []
            base_errs = []
            widths = []
            base_widths = []
            trained = []
            for bucket, info in buckets.items():
                tr = info.get("train") or {}
                n = tr.get("n", 0)
                if n < min_rows_per_bucket:
                    continue
                target = float(tr.get("target", 0.0))
                best_cover = float(tr.get("best_cover", 0.0))
                base_cover = float(tr.get("base_cover", 0.0))
                best_w = float(tr.get("best_width", 0.0))
                base_w = float(tr.get("base_width", 0.0))
                errs.append(abs(best_cover - target))
                base_errs.append(abs(base_cover - target))
                widths.append(best_w)
                base_widths.append(base_w)
                trained.append(bucket)
            if not trained:
                print(f"    Level {lvl_key}: not enough data per bucket yet (min {min_rows_per_bucket}).")
                continue
            mean_err = sum(errs)/len(errs) if errs else 0.0
            mean_base_err = sum(base_errs)/len(base_errs) if base_errs else 0.0
            mean_w = sum(widths)/len(widths) if widths else 0.0
            mean_base_w = sum(base_widths)/len(base_widths) if base_widths else 0.0
            delta_err = mean_base_err - mean_err
            delta_w = mean_w - mean_base_w
            sign = "+" if delta_err >= 0 else ""
            print(f"    Level {lvl_key}: trained={trained} | avg |cover-target| {mean_err:.3f} (base {mean_base_err:.3f}, {sign}{delta_err:.3f} better)")
            print(f"              avg width {mean_w:.3f} (base {mean_base_w:.3f}, {delta_w:+.3f})")
        print("")
    print("Tip: Match count uses unique match_id (best). If match_id is reused, your progress will look stuck.\n")
    return calibration_out


if __name__ == "__main__":
    train()