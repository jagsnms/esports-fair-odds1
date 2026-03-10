"""
Export identity + predictions for the most inverted (game, map) group.

Reads out/inversion_auc.json (from check_round_inversion_auc.py --by_map --out_json)
or re-computes per-map AUC, picks the worst (lowest AUC) group meeting --min_rows,
then writes out/inverted_map_witness.csv with game, map, round, y, p_unshaped,
score_raw, p_hat_final, rails, corridor_fraction, weight_profile, and team identity fields.

Usage (repo root):
  python tools/dump_inverted_map_witness.py
  python tools/dump_inverted_map_witness.py --inversion_json out/inversion_auc.json --min_rows 20
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def _build_labels(label_path: Path) -> dict[tuple[int, int, int], int]:
    """Build labels[(game_number, map_index, round_number)] = 1 if team_a wins else 0."""
    labels: dict[tuple[int, int, int], int] = {}
    for obj in _read_jsonl(label_path):
        ev = obj.get("event") if isinstance(obj.get("event"), dict) else None
        if not ev or ev.get("event_type") != "round_result":
            continue
        gn = ev.get("game_number")
        mi = ev.get("map_index") or obj.get("map_index")
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
    return labels


def _compute_auc(y: list[int], pred: list[float]) -> float | None:
    """AUC = (S - n1*(n1+1)/2) / (n1*n0); ranks by pred ascending."""
    n = len(y)
    if n == 0:
        return None
    n1 = sum(1 for v in y if v == 1)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return None
    order = sorted(range(n), key=lambda i: (pred[i], i))
    rank = [0.0] * n
    for r, idx in enumerate(order, start=1):
        rank[idx] = float(r)
    S = sum(rank[i] for i in range(n) if y[i] == 1)
    auc = (S - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return max(0.0, min(1.0, auc))


def _gather_rows(
    score_path: Path,
    labels: dict[tuple[int, int, int], int],
    phase: str,
    map_index: int | None,
    game_number: int | None,
) -> list[tuple[float, int, int, int, int]]:
    """Return list of (p_unshaped, y, game_number, map_index, round_number)."""
    rows: list[tuple[float, int, int, int, int]] = []
    for obj in _read_jsonl(score_path):
        if obj.get("schema") != "score_diag_v2":
            continue
        if obj.get("phase") != phase:
            continue
        gn = obj.get("game_number")
        mi = obj.get("map_index")
        rn = obj.get("round_number")
        pu = obj.get("p_unshaped")
        if mi is None or rn is None or pu is None:
            continue
        try:
            gn = int(gn) if gn is not None else 0
            mi = int(mi)
            rn = int(rn)
            pu = float(pu)
        except (TypeError, ValueError):
            continue
        if game_number is not None and gn != game_number:
            continue
        if map_index is not None and mi != map_index:
            continue
        key = (gn, mi, rn)
        if key not in labels:
            continue
        rows.append((pu, labels[key], gn, mi, rn))
    return rows


def _get_worst_map(
    inversion_path: Path,
    score_path: Path,
    label_path: Path,
    phase: str,
    min_rows: int,
) -> tuple[int, int] | None:
    """
    Return (game_number, map_index) for worst inverted map.
    If inversion_path exists and has per_map, use it; else re-compute per-map AUC.
    """
    if inversion_path.exists():
        try:
            with open(inversion_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    per_map = data.get("per_map")
    if isinstance(per_map, list) and len(per_map) > 0:
        # Use pre-computed per_map; pick lowest AUC with n_rows >= min_rows
        candidates = [m for m in per_map if m.get("n_rows", 0) >= min_rows]
        if not candidates:
            return None
        worst = min(candidates, key=lambda m: m["auc_p_unshaped"])
        return (int(worst["game_number"]), int(worst["map_index"]))

    # Re-compute per-map AUC
    if not label_path.exists() or not score_path.exists():
        return None
    labels = _build_labels(label_path)
    rows = _gather_rows(score_path, labels, phase, map_index=None, game_number=None)
    by_key: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for pu, yi, gn, mi, _ in rows:
        by_key[(gn, mi)].append((pu, yi))
    candidates = []
    for (gn, mi), pairs in by_key.items():
        if len(pairs) < min_rows:
            continue
        y_m = [p[1] for p in pairs]
        pred_m = [p[0] for p in pairs]
        auc_m = _compute_auc(y_m, pred_m)
        if auc_m is not None:
            candidates.append((gn, mi, auc_m, len(pairs)))
    if not candidates:
        return None
    worst = min(candidates, key=lambda x: (x[2], -x[3]))
    return (worst[0], worst[1])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export witness CSV for the most inverted (game, map) from AUC tool"
    )
    ap.add_argument(
        "--inversion_json",
        default="out/inversion_auc.json",
        help="JSON from check_round_inversion_auc.py --by_map --out_json",
    )
    ap.add_argument(
        "--label_input",
        default="logs/history_points.jsonl",
        help="Label JSONL (used when re-computing or from JSON path)",
    )
    ap.add_argument(
        "--score_input",
        default="logs/history_score_points.jsonl",
        help="Score diag JSONL (used when re-computing or from JSON path)",
    )
    ap.add_argument(
        "--phase",
        default="IN_PROGRESS",
        help="Phase filter for score rows",
    )
    ap.add_argument(
        "--min_rows",
        type=int,
        default=20,
        help="Min rows for a (game, map) to be considered",
    )
    ap.add_argument(
        "--out_csv",
        default="out/inverted_map_witness.csv",
        help="Output witness CSV path",
    )
    args = ap.parse_args()

    inversion_path = Path(args.inversion_json)
    label_path = Path(args.label_input)
    score_path = Path(args.score_input)
    phase = (args.phase or "IN_PROGRESS").strip()

    # Resolve paths from inversion JSON if present
    if inversion_path.exists():
        try:
            with open(inversion_path, "r", encoding="utf-8") as f:
                inv_data = json.load(f)
            if isinstance(inv_data.get("score_input"), str):
                p = Path(inv_data["score_input"])
                if p.exists():
                    score_path = p
            if isinstance(inv_data.get("label_input"), str):
                p = Path(inv_data["label_input"])
                if p.exists():
                    label_path = p
        except (json.JSONDecodeError, OSError):
            pass

    worst = _get_worst_map(
        inversion_path,
        score_path,
        label_path,
        phase,
        args.min_rows,
    )
    if worst is None:
        print("No (game, map) meeting min_rows found. Run check_round_inversion_auc.py --by_map --out_json first.")
        return

    game_number, map_index = worst
    labels = _build_labels(label_path)
    if not score_path.exists():
        print(f"Score file not found: {score_path}")
        return

    # Gather all score_diag_v2 rows for this (game, map)
    rows: list[dict[str, Any]] = []
    for obj in _read_jsonl(score_path):
        if obj.get("schema") != "score_diag_v2":
            continue
        if obj.get("phase") != phase:
            continue
        gn = obj.get("game_number")
        mi = obj.get("map_index")
        rn = obj.get("round_number")
        if gn is not None and mi is not None and rn is not None:
            try:
                gn = int(gn)
                mi = int(mi)
                rn = int(rn)
            except (TypeError, ValueError):
                continue
        else:
            continue
        if gn != game_number or mi != map_index:
            continue
        key = (gn, mi, rn)
        y = labels.get(key)
        if y is None:
            continue

        rail_low = obj.get("rail_low")
        rail_high = obj.get("rail_high")
        p_hat_final = obj.get("p_hat_final")
        if rail_low is not None:
            try:
                rail_low = float(rail_low)
            except (TypeError, ValueError):
                rail_low = None
        if rail_high is not None:
            try:
                rail_high = float(rail_high)
            except (TypeError, ValueError):
                rail_high = None
        if p_hat_final is not None:
            try:
                p_hat_final = float(p_hat_final)
            except (TypeError, ValueError):
                p_hat_final = None

        corridor_fraction = ""
        if (
            rail_low is not None
            and rail_high is not None
            and p_hat_final is not None
            and abs(rail_high - rail_low) > 1e-12
        ):
            corridor_fraction = (p_hat_final - rail_low) / (rail_high - rail_low)
            corridor_fraction = max(0.0, min(1.0, corridor_fraction))

        row: dict[str, Any] = {
            "game": gn,
            "map": mi,
            "round": rn,
            "y": y,
            "p_unshaped": obj.get("p_unshaped"),
            "score_raw": obj.get("score_raw"),
            "p_hat_final": p_hat_final,
            "rail_low": rail_low,
            "rail_high": rail_high,
            "corridor_fraction": corridor_fraction if corridor_fraction != "" else "",
            "weight_profile": obj.get("weight_profile", ""),
        }
        for k in ("team_one_id", "team_two_id", "team_one_provider_id", "team_two_provider_id", "team_a_is_team_one", "a_side"):
            row[k] = obj.get(k, "")
        for k, v in obj.items():
            if k.startswith("team_") and k not in row:
                row[k] = v
        rows.append(row)

    if not rows:
        print(f"No joined rows for game={game_number} map={map_index}. Check label/score inputs.")
        return

    # Column order: fixed first, then team_* and a_side
    fixed_cols = [
        "game", "map", "round", "y",
        "p_unshaped", "score_raw", "p_hat_final",
        "rail_low", "rail_high", "corridor_fraction",
        "weight_profile",
        "team_one_id", "team_two_id", "team_one_provider_id", "team_two_provider_id",
        "team_a_is_team_one", "a_side",
    ]
    extra_team = sorted(k for k in rows[0].keys() if k.startswith("team_") and k not in fixed_cols)
    fieldnames = fixed_cols + extra_team

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out_row = {}
            for fn in fieldnames:
                val = r.get(fn)
                if val is None:
                    val = ""
                elif isinstance(val, float):
                    if val == int(val):
                        val = int(val)
                out_row[fn] = val
            w.writerow(out_row)

    print(f"Worst inverted (game, map): game={game_number} map={map_index}  n_rows={len(rows)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
