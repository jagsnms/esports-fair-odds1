"""
Unit tests for tools/score_term_scale_report.py.
Verifies: script runs on synthetic score + label JSONL; produces CSV and summary JSON;
reconstruction error is tiny; term_loadout dominance in score space shows in rms_share.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_score_term_scale_report_runs_and_shows_term_dominance(tmp_path: object) -> None:
    """With synthetic label + score JSONL, report runs and term_loadout has highest rms_share."""
    base = Path(tmp_path)
    label_path = base / "history_points.jsonl"
    score_path = base / "history_score_points.jsonl"

    # One round_result so we have one label (0, 0, 1) -> 1
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": {
                        "event_type": "round_result",
                        "game_number": 0,
                        "map_index": 0,
                        "round_number": 1,
                        "round_winner_is_team_a": True,
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # One score line: (0,0,1), term_loadout large so it dominates in score space
    term_alive = 0.01
    term_hp = 0.02
    term_loadout = 0.50  # dominant
    term_bomb = 0.0
    term_cash = 0.0
    score_raw = term_alive + term_hp + term_loadout + term_bomb + term_cash
    with open(score_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "schema": "score_diag_v1",
                    "ts_ms": 1000000,
                    "t": 1000.0,
                    "game_number": 0,
                    "map_index": 0,
                    "round_number": 1,
                    "phase": "IN_PROGRESS",
                    "clamp_reason": None,
                    "p_hat_final": 0.55,
                    "rail_low": 0.4,
                    "rail_high": 0.6,
                    "score_raw": score_raw,
                    "p_unshaped": 0.54,
                    "term_contribs": {
                        "term_alive": term_alive,
                        "term_hp": term_hp,
                        "term_loadout": term_loadout,
                        "term_bomb": term_bomb,
                        "term_cash": term_cash,
                    },
                    "base_intercept": 0.0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    out_dir = base / "out"
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "tools" / "score_term_scale_report.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--score_input", str(score_path),
            "--label_input", str(label_path),
            "--out_dir", str(out_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    summary_path = out_dir / "score_term_scale_report_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["n_ticks_used"] == 1
    assert summary["reconstruction_max_abs_err"] < 1e-5
    assert summary["n_labels"] == 1

    csv_path = out_dir / "score_term_scale_report_terms.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) >= 2  # header + at least one term row
    header = lines[0]
    assert "term" in header and "rms_share" in header
    # Parse CSV: find term_loadout row and check it has highest rms_share
    cols = [c.strip() for c in header.split(",")]
    term_idx = cols.index("term")
    rms_share_idx = cols.index("rms_share")
    term_rms_shares = {}
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) > max(term_idx, rms_share_idx):
            term_rms_shares[parts[term_idx]] = float(parts[rms_share_idx])
    assert "term_loadout" in term_rms_shares
    loadout_share = term_rms_shares["term_loadout"]
    for t, share in term_rms_shares.items():
        assert loadout_share >= share - 1e-6, f"term_loadout rms_share {loadout_share} should be >= {t} {share}"


def test_score_term_scale_report_v2_reconstruction_near_zero(tmp_path: object) -> None:
    """With v2 schema (contrib_sum + residual_contrib), reconstruction_max_abs_err is ~0."""
    base = Path(tmp_path)
    label_path = base / "history_points.jsonl"
    score_path = base / "history_score_points.jsonl"

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": {
                        "event_type": "round_result",
                        "game_number": 0,
                        "map_index": 0,
                        "round_number": 1,
                        "round_winner_is_team_a": False,
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    term_alive = 0.02
    term_hp = -0.01
    term_loadout = 0.005
    term_bomb = 0.0
    term_cash = 0.0
    contrib_sum = term_alive + term_hp + term_loadout + term_bomb + term_cash
    residual_contrib = 0.012  # e.g. armor
    score_raw = contrib_sum + residual_contrib
    with open(score_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "schema": "score_diag_v2",
                    "ts_ms": 2000000,
                    "t": 2000.0,
                    "game_number": 0,
                    "map_index": 0,
                    "round_number": 1,
                    "phase": "IN_PROGRESS",
                    "clamp_reason": None,
                    "p_hat_final": 0.48,
                    "score_raw": score_raw,
                    "term_contribs": {
                        "term_alive": term_alive,
                        "term_hp": term_hp,
                        "term_loadout": term_loadout,
                        "term_bomb": term_bomb,
                        "term_cash": term_cash,
                    },
                    "base_intercept": 0.0,
                    "contrib_sum": contrib_sum,
                    "residual_contrib": residual_contrib,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    out_dir = base / "out"
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "tools" / "score_term_scale_report.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--score_input", str(score_path),
            "--label_input", str(label_path),
            "--out_dir", str(out_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    summary_path = out_dir / "score_term_scale_report_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["n_ticks_used"] == 1
    assert summary["reconstruction_max_abs_err"] < 1e-6
    assert "residual_mean_abs" in summary
    assert "residual_max_abs" in summary
