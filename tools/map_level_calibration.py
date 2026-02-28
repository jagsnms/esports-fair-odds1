"""
Map-level calibration from history_points.jsonl.

Reads logs/history_points.jsonl, builds segment labels from segment_result events,
extracts non-event ticks with explain + seg in labeled segments, and outputs:
- out/map_calibration.json: summary counts, clamped rates, clamp attribution
- out/map_term_<name>.csv: per-term 20-quantile bin tables (mean term, win rate, mean p_hat)

No model changes. Run from repo root: python tools/map_level_calibration.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INPUT_JSONL = "logs/history_points.jsonl"
OUT_DIR = "out"
N_BINS = 20

# Only ticks with clamp_reason None or in COMPUTED_CLAMP_REASONS are used for calibration.
IGNORE_REASONS = {"no_source", "no_compute", "inter_map_break", "replay_loop", "passthrough"}
COMPUTED_CLAMP_REASONS = {"rail_low", "rail_high", "rails_collapsed"}


def _safe_float(x, default: float = 0.0) -> float:
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

    # 1) Build label_by_map_index from segment_result events (prefer map_index over seg)
    label_by_map_index: dict[int, bool] = {}
    for obj in lines:
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if not ev or ev.get("event_type") != "segment_result":
            continue
        mi = ev.get("map_index")
        if mi is None:
            mi = ev.get("segment_id")
        if mi is None:
            mi = obj.get("map_index") or obj.get("seg")
        if mi is not None:
            try:
                mi = int(mi)
            except (TypeError, ValueError):
                continue
            winner_a = ev.get("map_winner_is_team_a")
            if winner_a is not None:
                label_by_map_index[mi] = bool(winner_a)

    # 2) Collect non-event ticks with explain and map_index (or seg fallback) in label_by_map_index
    rows: list[dict] = []
    segment_result_count = 0
    for obj in lines:
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if ev and ev.get("event_type") == "segment_result":
            segment_result_count += 1
            continue
        explain = obj.get("explain") if isinstance(obj.get("explain"), dict) else None
        mi = obj.get("map_index")
        if mi is None:
            mi = obj.get("seg")
        if mi is not None:
            try:
                mi = int(mi)
            except (TypeError, ValueError):
                mi = None
        if not explain or mi is None or mi not in label_by_map_index:
            continue
        y = label_by_map_index[mi]
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
            "seg": obj.get("seg"),
        })

    # 3) Use only computed ticks: phase != "idle" AND clamp_reason None or in {rail_low, rail_high, rails_collapsed}
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
    n_maps_labeled = len(label_by_map_index)

    summary = {
        "n_lines": len(lines),
        "n_segment_result_events": segment_result_count,
        "n_maps_with_label": n_maps_labeled,
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

    # 4) Per-term bin tables (20 quantile bins) from used_rows only
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
        quantiles = np.percentile(arr, np.linspace(0, 100, N_BINS + 1))
        bin_lo = quantiles[:-1]
        bin_hi = quantiles[1:]
        table = []
        for i in range(N_BINS):
            if i < N_BINS - 1:
                mask = (arr >= bin_lo[i]) & (arr < bin_hi[i])
            else:
                mask = arr >= bin_lo[i]
            count = int(np.sum(mask))
            if count == 0:
                table.append({
                    "bin_lo": float(bin_lo[i]),
                    "bin_hi": float(bin_hi[i]),
                    "mean_term": float(bin_lo[i]),
                    "win_rate": 0.0,
                    "mean_p_hat": 0.5,
                    "count": 0,
                })
                continue
            mean_term = float(np.mean(arr[mask]))
            indices = np.where(mask)[0]
            ys = np.array([used_rows[j]["y"] for j in indices])
            mean_p_hats = np.array([used_rows[j]["p_hat"] for j in indices])
            win_rate = float(np.mean(ys))
            mean_p_hat = float(np.mean(mean_p_hats))
            table.append({
                "bin_lo": float(bin_lo[i]),
                "bin_hi": float(bin_hi[i]),
                "mean_term": mean_term,
                "win_rate": win_rate,
                "mean_p_hat": mean_p_hat,
                "count": count,
            })
        bin_tables[term_name] = table

    # 5) Clamp attribution: dominant abs component among q_terms + micro_adj by clamp side (used_rows only)
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
    clamp_attribution = {
        "rail_low": dict(components_by_side["rail_low"]),
        "rail_high": dict(components_by_side["rail_high"]),
    }
    summary["clamp_attribution"] = clamp_attribution

    # 6) Write out/map_calibration.json
    out_cal = {
        "summary": summary,
        "bin_table_terms": list(bin_tables.keys()),
    }
    cal_path = out_path / "map_calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(out_cal, f, indent=2)
    print(f"Wrote {cal_path}")

    # 7) Write out/map_term_<name>.csv
    for term_name in bin_tables:
        table = bin_tables[term_name]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in term_name)
        csv_path = out_path / f"map_term_{safe_name}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("bin_lo,bin_hi,mean_term,win_rate,mean_p_hat,count\n")
            for row in table:
                f.write(f"{row['bin_lo']},{row['bin_hi']},{row['mean_term']},{row['win_rate']},{row['mean_p_hat']},{row['count']}\n")
        print(f"Wrote {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
