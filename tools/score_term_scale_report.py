"""
Score-space term dominance report from logs/history_score_points.jsonl.

Uses round labels from logs/history_points.jsonl (round_result events) and joins to
score diagnostic lines by (game_number, map_index, round_number). Reports per-term
stats (mean, std, RMS, percentiles, rms_share) in SCORE space so we can see whether
e.g. term_loadout dominates before asymptotic shaping.

Outputs: out/score_term_scale_report_terms.csv, out/score_term_scale_report_summary.json

Usage (repo root):
  python tools/score_term_scale_report.py
  python tools/score_term_scale_report.py --score_input logs/history_score_points.jsonl --phase IN_PROGRESS
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TERM_KEYS = ("term_alive", "term_hp", "term_loadout", "term_bomb", "term_cash")
SCORE_DIAG_SCHEMAS = ("score_diag_v1", "score_diag_v2")
DEFAULT_IGNORE_REASONS = "no_source,no_compute,inter_map_break,replay_loop,passthrough"


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_ignore_reasons(s: str) -> set[str]:
    if not s.strip():
        return set()
    return {r.strip() for r in s.split(",") if r.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score-space term dominance report from history_score_points.jsonl"
    )
    parser.add_argument(
        "--score_input",
        default="logs/history_score_points.jsonl",
        help="Score diagnostics JSONL path",
    )
    parser.add_argument(
        "--label_input",
        default="logs/history_points.jsonl",
        help="Label source JSONL (round_result events)",
    )
    parser.add_argument(
        "--phase",
        default="IN_PROGRESS",
        help="Phase filter (default IN_PROGRESS); use empty string to include all non-idle",
    )
    parser.add_argument(
        "--ignore_reasons",
        default=DEFAULT_IGNORE_REASONS,
        help="Comma-separated clamp_reason values to exclude",
    )
    parser.add_argument("--out_dir", default="out", help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    score_path = repo_root / args.score_input
    label_path = repo_root / args.label_input
    out_path = repo_root / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    ignore_reasons = _parse_ignore_reasons(args.ignore_reasons)
    phase_filter: str | None = args.phase.strip() if args.phase and args.phase.strip() else None

    # 1) Build labels from label_input (history_points.jsonl)
    label_lines = _read_jsonl(label_path)
    labels: dict[tuple[int, int, int], int] = {}
    n_event_round_results = 0
    for obj in label_lines:
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if not ev or ev.get("event_type") != "round_result":
            continue
        n_event_round_results += 1
        gn = ev.get("game_number")
        mi = ev.get("map_index")
        if mi is None:
            mi = obj.get("map_index")
        rn = ev.get("round_number")
        if rn is None or mi is None:
            continue
        try:
            gn = int(gn) if gn is not None else 0
            mi = int(mi)
            rn = int(rn)
        except (TypeError, ValueError):
            continue
        winner_a = ev.get("round_winner_is_team_a")
        if winner_a is not None:
            labels[(gn, mi, rn)] = 1 if winner_a else 0

    n_labels = len(labels)

    # 2) Scan score_input, filter, join to labels, extract score_raw and term_contribs (v1 or v2)
    score_lines = _read_jsonl(score_path)
    n_score_lines = len(score_lines)
    skipped = defaultdict(int)
    term_arrays: dict[str, list[float]] = defaultdict(list)
    score_raw_list: list[float] = []
    sum_contribs_list: list[float] = []
    reconstruction_errors: list[float] = []
    residual_list: list[float] = []  # v2 only: residual_contrib per row

    for obj in score_lines:
        schema = obj.get("schema")
        if schema not in SCORE_DIAG_SCHEMAS:
            skipped["wrong_schema"] += 1
            continue
        phase = obj.get("phase")
        if phase == "idle":
            skipped["idle_phase"] += 1
            continue
        if phase_filter is not None and phase != phase_filter:
            skipped["phase_filter"] += 1
            continue
        clamp_reason = obj.get("clamp_reason")
        if clamp_reason is not None and clamp_reason in ignore_reasons:
            skipped["clamp_ignored"] += 1
            continue
        gn = obj.get("game_number")
        mi = obj.get("map_index")
        rn = obj.get("round_number")
        if mi is None or rn is None:
            skipped["missing_key"] += 1
            continue
        try:
            gn = int(gn) if gn is not None else 0
            mi = int(mi)
            rn = int(rn)
        except (TypeError, ValueError):
            skipped["missing_key"] += 1
            continue
        key = (gn, mi, rn)
        if key not in labels:
            skipped["no_label"] += 1
            continue
        score_raw = obj.get("score_raw")
        if score_raw is None:
            skipped["missing_score_raw"] += 1
            continue
        term_contribs = obj.get("term_contribs")
        if not isinstance(term_contribs, dict):
            skipped["missing_term_contribs"] += 1
            continue

        score_raw_val = _safe_float(score_raw)
        score_raw_list.append(score_raw_val)
        sum_contrib = 0.0
        for k in TERM_KEYS:
            v = _safe_float(term_contribs.get(k))
            term_arrays[k].append(v)
            sum_contrib += v
        sum_contribs_list.append(sum_contrib)

        if schema == "score_diag_v2":
            contrib_sum = _safe_float(obj.get("contrib_sum"), sum_contrib)
            residual_contrib = _safe_float(obj.get("residual_contrib"))
            residual_list.append(residual_contrib)
            # Reconstruction: score_raw ≈ contrib_sum + residual_contrib (exact for v2)
            recon_err = abs(score_raw_val - (contrib_sum + residual_contrib))
            reconstruction_errors.append(recon_err)
        else:
            # v1: no residual; reconstruction err = score_raw - sum(term_contribs)
            reconstruction_errors.append(abs(score_raw_val - sum_contrib))

    n_ticks_used = len(score_raw_list)

    # 3) Per-term stats: n, mean, std, RMS, percentiles (signed + abs), rms_share
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    rows: list[dict[str, Any]] = []
    rms_by_term: dict[str, float] = {}

    for term_name in sorted(term_arrays.keys()):
        arr = np.array(term_arrays[term_name], dtype=float)
        n = len(arr)
        if n == 0:
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr)) if n > 1 else 0.0
        rms = float(np.sqrt(np.mean(arr ** 2)))
        rms_by_term[term_name] = rms

        pct_signed = np.percentile(arr, percentiles)
        arr_abs = np.abs(arr)
        pct_abs = np.percentile(arr_abs, percentiles)

        row = {
            "term": term_name,
            "n": n,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "rms": round(rms, 6),
        }
        for i, p in enumerate(percentiles):
            row[f"p{p:02d}"] = round(float(pct_signed[i]), 6)
        for i, p in enumerate(percentiles):
            row[f"abs_p{p:02d}"] = round(float(pct_abs[i]), 6)
        row["rms_share"] = None
        rows.append(row)

    sum_rms = sum(rms_by_term.values()) if rms_by_term else 0.0
    for row in rows:
        rms = rms_by_term.get(row["term"], 0.0)
        row["rms_share"] = round(rms / sum_rms, 6) if sum_rms > 0 else 0.0

    # Reconstruction check
    max_abs_err = float(np.max(reconstruction_errors)) if reconstruction_errors else 0.0
    mean_abs_err = float(np.mean(reconstruction_errors)) if reconstruction_errors else 0.0
    # v2 residual stats (when we had v2 rows)
    residual_mean_abs = float(np.mean(np.abs(residual_list))) if residual_list else None
    residual_max_abs = float(np.max(np.abs(residual_list))) if residual_list else None

    # Write CSV
    csv_columns = [
        "term", "n", "mean", "std", "rms",
        "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
        "abs_p01", "abs_p05", "abs_p10", "abs_p25", "abs_p50", "abs_p75", "abs_p90", "abs_p95", "abs_p99",
        "rms_share",
    ]
    csv_path = out_path / "score_term_scale_report_terms.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in csv_columns) + "\n")
    print(f"Wrote {csv_path}")

    # Write summary JSON
    summary: dict[str, Any] = {
        "score_input_path": str(score_path),
        "label_input_path": str(label_path),
        "phase_filter": phase_filter,
        "ignore_reasons": list(sorted(ignore_reasons)),
        "n_score_lines": n_score_lines,
        "n_event_round_results": n_event_round_results,
        "n_labels": n_labels,
        "n_ticks_used": n_ticks_used,
        "skipped_counts": dict(skipped),
        "sum_rms_total": round(sum_rms, 6),
        "reconstruction_max_abs_err": round(max_abs_err, 10),
        "reconstruction_mean_abs_err": round(mean_abs_err, 10),
    }
    if residual_list:
        summary["residual_mean_abs"] = round(residual_mean_abs, 10)
        summary["residual_max_abs"] = round(residual_max_abs, 10)
    summary_path = out_path / "score_term_scale_report_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
