"""
Unit tests for tools/fit_midround_weights_score_diag.py.
Verifies: script runs and produces JSON, CSV, calibration CSV with expected structure.
When default logs have score_diag_v2 + round_result data, runs full check; else checks output schema only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fit_midround_weights_produces_outputs() -> None:
    """Run script on default paths; if data exists verify outputs and schema; else skip (no failure)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "tools" / "fit_midround_weights_score_diag.py"
    out_subdir = repo_root / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out_dir", "out",
            "--seed", "42",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    # Script exits 0 even when no rows (prints message and returns)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    json_path = out_subdir / "midround_fit_weights.json"
    if not json_path.exists():
        # No data or no rows after filtering - skip structure checks
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "n_rows" in data and data["n_rows"] >= 1
    assert data["feature_names"] == ["alive", "hp", "loadout", "bomb"]
    assert "coef_std" in data and "coef_unstd" in data
    assert "current_term_coef" in data and "suggested_coef" in data
    assert "metrics_train" in data and "auc" in data["metrics_train"]
    assert "metrics_test" in data and "brier" in data["metrics_test"]
    assert "scale_g" in data

    csv_path = out_subdir / "midround_fit_weights.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "feature,coef_unstd,coef_std,odds_ratio_per_1std,current_coef,suggested_coef"
    assert len(lines) >= 5  # header + 4 features

    cal_path = out_subdir / "midround_fit_calibration_bins.csv"
    assert cal_path.exists()
    cal_lines = cal_path.read_text(encoding="utf-8").strip().split("\n")
    assert cal_lines[0] == "bin_lo,bin_hi,mean_pred,mean_actual,count"


def test_fit_midround_weights_auc_uses_positive_class() -> None:
    """When script runs on default logs, AUC must be > 0.5 (P(y=1) used; direction sanity says y=1 has higher means)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "tools" / "fit_midround_weights_score_diag.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--out_dir", "out", "--seed", "42"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    json_path = repo_root / "out" / "midround_fit_weights.json"
    if not json_path.exists():
        return  # no data; skip AUC assertion
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    auc_test = data.get("metrics_test", {}).get("auc")
    if auc_test is None:
        return
    assert auc_test > 0.5, (
        f"AUC should be > 0.5 when using P(y=1) (direction sanity shows y=1 has higher means). Got {auc_test}"
    )
