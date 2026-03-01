"""
Term-scale diagnostic for labeled round ticks (CS2 history JSONL).

Schema: compute ticks have explain.q_terms, explain.micro_adj, explain.final (p_hat_final, clamp_reason);
event lines have event_type "round_result" with round_winner_is_team_a, map_index, round_number, game_number.

Builds round labels from round_result events, joins labels onto compute ticks by (game_number, map_index, round_number),
filters out idle + non-compute clamp reasons, then for each term computes:
  N, mean, std, RMS, percentiles (signed and abs), and RMS share.

Outputs: out/term_scale_report_terms.csv, out/term_scale_report_summary.json

Usage (repo root):
  python tools/term_scale_report.py
  python tools/term_scale_report.py --input logs/history_points.jsonl --phase IN_PROGRESS
  python tools/term_scale_report.py --include_terms q_terms,micro_adj --ignore_reasons no_source,no_compute
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def _parse_include_terms(s: str) -> list[str]:
    if not s.strip():
        return ["q_terms", "micro_adj"]
    return [t.strip() for t in s.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Term-scale diagnostic for labeled round ticks")
    parser.add_argument("--input", default="logs/history_points.jsonl", help="Input JSONL path")
    parser.add_argument("--phase", default=None, help="Optional phase filter (e.g. IN_PROGRESS); omit to include all non-idle")
    parser.add_argument("--eps", type=float, default=None, help="Reserved for future pegged checks")
    parser.add_argument(
        "--include_terms",
        default="q_terms,micro_adj",
        help="Comma-separated: q_terms and/or micro_adj",
    )
    parser.add_argument(
        "--ignore_reasons",
        default="no_source,no_compute,inter_map_break,replay_loop,passthrough",
        help="Comma-separated clamp_reason values to exclude",
    )
    parser.add_argument("--out_dir", default="out", help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    input_path = repo_root / args.input
    out_path = repo_root / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    ignore_reasons = _parse_ignore_reasons(args.ignore_reasons)
    include_groups = _parse_include_terms(args.include_terms)
    phase_filter: str | None = args.phase.strip() if args.phase and args.phase.strip() else None

    lines = _read_jsonl(input_path)
    n_lines = len(lines)

    # Pass 1: build labels (game_number, map_index, round_number) -> y (1 = team A wins)
    labels: dict[tuple[int, int, int], int] = {}
    n_event_round_results = 0
    for obj in lines:
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

    # Pass 2: collect term values from compute ticks that pass filters
    skipped = defaultdict(int)
    n_ticks_used = 0
    term_arrays: dict[str, list[float]] = defaultdict(list)  # term -> list of values
    term_group: dict[str, str] = {}  # term -> "q_terms" | "micro_adj"

    for obj in lines:
        if obj.get("event") is not None and isinstance(obj.get("event"), dict):
            continue  # skip event lines
        explain = obj.get("explain") if isinstance(obj.get("explain"), dict) else None
        if not explain:
            skipped["no_explain"] += 1
            continue
        final = explain.get("final")
        if not isinstance(final, dict):
            skipped["no_final"] += 1
            continue
        phase = explain.get("phase")
        if phase == "idle":
            skipped["idle_phase"] += 1
            continue
        if phase_filter is not None and phase != phase_filter:
            skipped["phase_filter"] += 1
            continue
        clamp_reason = final.get("clamp_reason")
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

        n_ticks_used += 1
        q_terms = explain.get("q_terms") if isinstance(explain.get("q_terms"), dict) else {}
        micro_adj = explain.get("micro_adj") if isinstance(explain.get("micro_adj"), dict) else {}

        for group in include_groups:
            if group == "q_terms":
                term_dict = q_terms
            elif group == "micro_adj":
                term_dict = micro_adj
            else:
                continue
            for term_name, raw_val in term_dict.items():
                v = _safe_float(raw_val)
                term_arrays[term_name].append(v)
                term_group[term_name] = group


    # Per-term stats: mean, std, RMS, percentiles (signed + abs), rms_share
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
            "group": term_group.get(term_name, ""),
            "n": n,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "rms": round(rms, 6),
        }
        for i, p in enumerate(percentiles):
            row[f"p{p:02d}"] = round(float(pct_signed[i]), 6)
        for i, p in enumerate(percentiles):
            row[f"abs_p{p:02d}"] = round(float(pct_abs[i]), 6)
        row["rms_share"] = None  # fill after sum_rms
        rows.append(row)

    sum_rms = sum(rms_by_term.values()) if rms_by_term else 0.0
    for row in rows:
        rms = rms_by_term.get(row["term"], 0.0)
        row["rms_share"] = round(rms / sum_rms, 6) if sum_rms > 0 else 0.0

    # Per-group sum_rms for summary
    sum_rms_by_group: dict[str, float] = {}
    for term_name, rms in rms_by_term.items():
        g = term_group.get(term_name, "")
        sum_rms_by_group[g] = sum_rms_by_group.get(g, 0.0) + rms

    # Write CSV
    csv_columns = [
        "term", "group", "n", "mean", "std", "rms",
        "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
        "abs_p01", "abs_p05", "abs_p10", "abs_p25", "abs_p50", "abs_p75", "abs_p90", "abs_p95", "abs_p99",
        "rms_share",
    ]
    csv_path = out_path / "term_scale_report_terms.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in csv_columns) + "\n")
    print(f"Wrote {csv_path}")

    # Write summary JSON
    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "phase_filter": phase_filter,
        "ignore_reasons": list(sorted(ignore_reasons)),
        "include_terms_groups": include_groups,
        "n_lines": n_lines,
        "n_event_round_results": n_event_round_results,
        "n_labels": n_labels,
        "n_ticks_used": n_ticks_used,
        "skipped_counts": dict(skipped),
        "sum_rms_by_group": sum_rms_by_group,
        "sum_rms_total": round(sum_rms, 6),
    }
    if args.eps is not None:
        summary["eps"] = args.eps
    summary_path = out_path / "term_scale_report_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
