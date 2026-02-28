"""
Round-level calibration from history_points.jsonl.

Reads logs/history_points.jsonl, builds round labels from round_result events
keyed by (map_index, round_number), extracts non-event ticks with explain
and (map_index, round_number) in label_round, and outputs:
- out/round_calibration.json: summary (incl. y_count, y_rate, mean_p_hat_y0/y1, mean_term_*_y0/y1), clamp attribution, label coverage
- out/round_term_<name>.csv: per-term bin tables (smart n_bins from n_unique), non-empty bins only

Same exclude rules as map-level: phase != "idle", clamp_reason None or in {rail_low, rail_high, rails_collapsed}.
Run from repo root: python tools/round_level_calibration.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INPUT_JSONL = "logs/history_points.jsonl"
OUT_DIR = "out"
N_BINS_MAX = 20
N_BINS_MIN = 5

IGNORE_REASONS = {"no_source", "no_compute", "inter_map_break", "replay_loop", "passthrough"}
COMPUTED_CLAMP_REASONS = {"rail_low", "rail_high", "rails_collapsed"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
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


def run_calibration(lines: list[dict]) -> tuple[dict, list[dict], dict[str, list[dict]]]:
    """Build round labels, collect rows, filter to used_rows, compute summary and bin_tables. Returns (summary, used_rows, bin_tables)."""
    # 1) Build label_round from round_result events: (map_index, round_number) -> round_winner_is_team_a
    label_round: dict[tuple[int, int], bool] = {}
    for obj in lines:
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if not ev or ev.get("event_type") != "round_result":
            continue
        rn = ev.get("round_number")
        mi = ev.get("map_index")
        if mi is None:
            mi = obj.get("map_index")
        if rn is None or mi is None:
            continue
        try:
            mi, rn = int(mi), int(rn)
        except (TypeError, ValueError):
            continue
        winner_a = ev.get("round_winner_is_team_a")
        if winner_a is not None:
            label_round[(mi, rn)] = bool(winner_a)

    # 2) Collect non-event ticks with explain and (map_index, round_number) in label_round
    rows: list[dict] = []
    round_result_count = 0
    for obj in lines:
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if ev and ev.get("event_type") == "round_result":
            round_result_count += 1
            continue
        explain = obj.get("explain") if isinstance(obj.get("explain"), dict) else None
        mi = obj.get("map_index")
        rn = obj.get("round_number")
        if mi is not None and rn is not None:
            try:
                mi, rn = int(mi), int(rn)
            except (TypeError, ValueError):
                mi, rn = None, None
        if not explain or mi is None or rn is None or (mi, rn) not in label_round:
            continue
        y = label_round[(mi, rn)]
        phase = explain.get("phase")
        final = explain.get("final") or {}
        p_hat = _safe_float(final.get("p_hat_final"), obj.get("p", 0.5))
        clamp_reason = final.get("clamp_reason")
        rails = explain.get("rails") or {}
        corridor_width = _safe_float(rails.get("corridor_width"))
        q_terms = explain.get("q_terms")
        if not isinstance(q_terms, dict):
            q_terms = {}
        micro_adj = explain.get("micro_adj")
        if not isinstance(micro_adj, dict):
            micro_adj = {}
        rows.append({
            "y": int(y),
            "p_hat": p_hat,
            "phase": phase,
            "clamp_reason": clamp_reason,
            "corridor_width": corridor_width,
            "q_terms": {k: _safe_float(v) for k, v in q_terms.items()},
            "micro_adj": {k: _safe_float(v) for k, v in micro_adj.items()},
            "map_index": mi,
            "round_number": rn,
        })

    # 3) Use only computed ticks: phase != "idle" AND clamp_reason None or in COMPUTED_CLAMP_REASONS
    used_rows = [
        r for r in rows
        if r.get("phase") != "idle"
        and (r.get("clamp_reason") is None or r.get("clamp_reason") in COMPUTED_CLAMP_REASONS)
    ]
    excluded = [r for r in rows if r not in used_rows]
    ignored_by_phase = Counter((r.get("phase") or "null") for r in excluded)
    ignored_by_clamp_reason = Counter((r.get("clamp_reason") or "null") for r in excluded)

    n_used_ticks = len(used_rows)
    n_clamped = sum(1 for r in used_rows if r.get("clamp_reason") in COMPUTED_CLAMP_REASONS)
    n_clamp_low = sum(1 for r in used_rows if r.get("clamp_reason") == "rail_low")
    n_clamp_high = sum(1 for r in used_rows if r.get("clamp_reason") == "rail_high")
    n_clamp_collapsed = sum(1 for r in used_rows if r.get("clamp_reason") == "rails_collapsed")
    clamped_rate = n_clamped / n_used_ticks if n_used_ticks else 0.0
    n_rounds_labeled = len(label_round)

    summary = {
        "n_lines": len(lines),
        "n_round_result_events": round_result_count,
        "n_rounds_with_label": n_rounds_labeled,
        "n_labeled_ticks": len(rows),
        "n_used_ticks": n_used_ticks,
        "ignored_by_phase": dict(ignored_by_phase),
        "ignored_by_clamp_reason": dict(ignored_by_clamp_reason),
        "n_clamped": n_clamped,
        "n_clamp_low": n_clamp_low,
        "n_clamp_high": n_clamp_high,
        "n_clamp_collapsed": n_clamp_collapsed,
        "clamped_rate": round(clamped_rate, 6),
    }

    # 4) Per-term bin tables from used_rows (smart n_bins: min(20, max(5, n_unique)), non-empty bins only)
    all_term_names: list[str] = []
    seen: set[str] = set()
    for r in used_rows:
        for k in (r.get("q_terms") or {}):
            if k not in seen:
                seen.add(k)
                all_term_names.append(k)
        for k in (r.get("micro_adj") or {}):
            if k not in seen:
                seen.add(k)
                all_term_names.append(k)

    bin_tables: dict[str, list[dict]] = {}
    for term_name in all_term_names:
        values = []
        for r in used_rows:
            v = (r.get("q_terms") or {}).get(term_name)
            if v is None:
                v = (r.get("micro_adj") or {}).get(term_name)
            values.append(_safe_float(v))
        arr = np.array(values)
        if arr.size == 0:
            bin_tables[term_name] = []
            continue
        arr = np.nan_to_num(arr, nan=0.0)
        n_unique = int(np.unique(arr).size)
        n_bins = min(N_BINS_MAX, max(N_BINS_MIN, n_unique))
        try:
            bin_labels = pd.qcut(arr, q=n_bins, duplicates="drop")
        except Exception:
            bin_labels = pd.qcut(arr, q=1)
        bin_indices = np.asarray(bin_labels.codes)  # 0..n_cats-1; -1 for NaN
        bin_indices = np.clip(bin_indices, 0, None)
        n_bins_used = len(bin_labels.categories)
        # Aggregate per bin (only non-empty)
        table = []
        for b in range(n_bins_used):
            mask = bin_indices == b
            count = int(np.sum(mask))
            if count == 0:
                continue
            indices = np.where(mask)[0]
            bin_vals = arr[mask]
            bin_lo = float(np.min(bin_vals))
            bin_hi = float(np.max(bin_vals))
            mean_term = float(np.mean(bin_vals))
            ys = np.array([used_rows[int(j)]["y"] for j in indices])
            mean_p_hats = np.array([used_rows[int(j)]["p_hat"] for j in indices])
            win_rate = float(np.mean(ys))
            mean_p_hat = float(np.mean(mean_p_hats))
            table.append({
                "bin_lo": bin_lo,
                "bin_hi": bin_hi,
                "mean_term": mean_term,
                "win_rate": win_rate,
                "mean_p_hat": mean_p_hat,
                "count": count,
            })
        # Assign consecutive bin_index and n_bins_used (only non-empty bins written)
        n_written = len(table)
        for i, row in enumerate(table):
            row["bin_index"] = i
            row["n_bins_used"] = n_written
        bin_tables[term_name] = table

    # 5) Clamp attribution
    components_by_side: dict[str, dict[str, int]] = {"rail_low": defaultdict(int), "rail_high": defaultdict(int)}
    for r in used_rows:
        reason = r.get("clamp_reason")
        if reason not in ("rail_low", "rail_high"):
            continue
        combined = {**(r.get("q_terms") or {}), **(r.get("micro_adj") or {})}
        if not combined:
            continue
        best_key = max(combined.keys(), key=lambda k: abs(combined[k]))
        components_by_side[reason][best_key] += 1
    summary["clamp_attribution"] = {
        "rail_low": dict(components_by_side["rail_low"]),
        "rail_high": dict(components_by_side["rail_high"]),
    }

    # Label coverage: (map_index, round_number) pairs that have at least one used tick
    label_coverage = len(set((r["map_index"], r["round_number"]) for r in used_rows))
    summary["label_coverage_rounds"] = label_coverage

    # Label balance and p_hat separation (over used ticks)
    y_counts = Counter(r["y"] for r in used_rows)
    summary["y_count"] = {str(k): v for k, v in sorted(y_counts.items())}
    n_used = len(used_rows)
    summary["y_rate"] = round(y_counts.get(1, 0) / n_used, 6) if n_used else 0.0
    if n_used:
        p_hats_y0 = [r["p_hat"] for r in used_rows if r["y"] == 0]
        p_hats_y1 = [r["p_hat"] for r in used_rows if r["y"] == 1]
        summary["mean_p_hat_y0"] = round(sum(p_hats_y0) / len(p_hats_y0), 6) if p_hats_y0 else None
        summary["mean_p_hat_y1"] = round(sum(p_hats_y1) / len(p_hats_y1), 6) if p_hats_y1 else None
    else:
        summary["mean_p_hat_y0"] = None
        summary["mean_p_hat_y1"] = None
    # Mean term value by outcome (for each term present in used_rows)
    for term_name in all_term_names:
        vals_y0 = []
        vals_y1 = []
        for r in used_rows:
            v = (r.get("q_terms") or {}).get(term_name)
            if v is None:
                v = (r.get("micro_adj") or {}).get(term_name)
            v = _safe_float(v)
            if r["y"] == 0:
                vals_y0.append(v)
            else:
                vals_y1.append(v)
        mn0 = round(sum(vals_y0) / len(vals_y0), 6) if vals_y0 else None
        mn1 = round(sum(vals_y1) / len(vals_y1), 6) if vals_y1 else None
        summary[f"mean_term_{term_name}_y0"] = mn0
        summary[f"mean_term_{term_name}_y1"] = mn1
    # Degenerate labels warning
    if len(y_counts) < 2:
        summary["warning"] = "Labels are constant; calibration curves not meaningful yet."

    return summary, used_rows, bin_tables


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    input_path = repo_root / INPUT_JSONL
    out_path = repo_root / OUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    lines = _read_jsonl(str(input_path))
    if not lines:
        print(f"No lines read from {input_path}")
        return

    summary, used_rows, bin_tables = run_calibration(lines)

    out_cal = {"summary": summary, "bin_table_terms": list(bin_tables.keys())}
    cal_path = out_path / "round_calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(out_cal, f, indent=2)
    print(f"Wrote {cal_path}")

    # 7) Write out/round_term_<name>.csv (non-empty bins only; bin_index, n_bins_used)
    for term_name in bin_tables:
        table = bin_tables[term_name]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in term_name)
        csv_path = out_path / f"round_term_{safe_name}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("bin_index,n_bins_used,bin_lo,bin_hi,mean_term,win_rate,mean_p_hat,count\n")
            for row in table:
                f.write(f"{row['bin_index']},{row['n_bins_used']},{row['bin_lo']},{row['bin_hi']},{row['mean_term']},{row['win_rate']},{row['mean_p_hat']},{row['count']}\n")
        print(f"Wrote {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
