"""
Runtime sanity check: correlation of p_unshaped vs p_hat_final on score diag logs.

The corridor mapping must be monotonic: higher p_unshaped => higher p_hat_final.
If corr(p_unshaped, p_hat_final) is negative, the mapping is inverted.

Usage (repo root):
  python tools/corridor_correlation_check.py
  python tools/corridor_correlation_check.py --score_input logs/history_score_points.jsonl --phase IN_PROGRESS
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Print corr(p_unshaped, p_hat_final) from score diag JSONL")
    ap.add_argument(
        "--score_input",
        default="logs/history_score_points.jsonl",
        help="Path to score diag JSONL (score_diag_v1/v2)",
    )
    ap.add_argument(
        "--phase",
        default="IN_PROGRESS",
        help="Filter by phase (default IN_PROGRESS); use '' to include all",
    )
    args = ap.parse_args()

    path = Path(args.score_input)
    if not path.exists():
        print(f"File not found: {path}")
        return

    phase_filter = (args.phase or "").strip()
    pu_list: list[float] = []
    pf_list: list[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if phase_filter and obj.get("phase") != phase_filter:
                continue
            pu = obj.get("p_unshaped")
            pf = obj.get("p_hat_final")
            if pu is None or pf is None:
                continue
            try:
                pu_list.append(float(pu))
                pf_list.append(float(pf))
            except (TypeError, ValueError):
                continue

    n = len(pu_list)
    if n < 2:
        print(f"Too few rows (n={n}) after filtering (phase={phase_filter!r}). Need at least 2.")
        return

    pu_arr = np.array(pu_list, dtype=float)
    pf_arr = np.array(pf_list, dtype=float)
    one_minus_pf = 1.0 - pf_arr

    corr_direct = float(np.corrcoef(pu_arr, pf_arr)[0, 1]) if np.std(pu_arr) > 1e-12 and np.std(pf_arr) > 1e-12 else float("nan")
    corr_inverted = float(np.corrcoef(pu_arr, one_minus_pf)[0, 1]) if np.std(pu_arr) > 1e-12 and np.std(one_minus_pf) > 1e-12 else float("nan")

    print(f"Rows (phase={phase_filter!r}): {n}")
    print(f"  corr(p_unshaped, p_hat_final)       = {corr_direct:.6f}")
    print(f"  corr(p_unshaped, 1 - p_hat_final)  = {corr_inverted:.6f}")
    if not np.isnan(corr_direct):
        if corr_direct < 0:
            print("  -> WARNING: mapping appears INVERTED (corr < 0). Fix corridor/rail order or mixture.")
        elif corr_direct > 0.3:
            print("  -> OK: positive correlation (monotonic mapping).")


if __name__ == "__main__":
    main()
