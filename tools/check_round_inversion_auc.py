"""
Detect "clean inversion": p_unshaped predicts opposite of round label y.

Uses AUC(p_unshaped vs y) and AUC(1 - p_unshaped vs y). If AUC is near 0 and
inverted AUC near 1, we have a true identity/sign mismatch (e.g. wrong team convention).

Usage (repo root):
  python tools/check_round_inversion_auc.py
  python tools/check_round_inversion_auc.py --by_map
  python tools/check_round_inversion_auc.py --map_index 0 --out_json out/inversion_auc.json
"""
from __future__ import annotations

import argparse
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
    """
    AUC = (S - n1*(n1+1)/2) / (n1*n0) where S = sum of ranks of positive class.
    Ranks by pred ascending (rank 1 = smallest). No sklearn.
    """
    n = len(y)
    if n == 0:
        return None
    n1 = sum(1 for v in y if v == 1)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return None
    # Sort by prediction (ascending); rank 1 = smallest
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
        y = labels[key]
        rows.append((pu, y, gn, mi, rn))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check for round-label inversion via AUC(p_unshaped vs y)"
    )
    ap.add_argument(
        "--label_input",
        default="logs/history_points.jsonl",
        help="JSONL with round_result events",
    )
    ap.add_argument(
        "--score_input",
        default="logs/history_score_points.jsonl",
        help="Score diag JSONL (score_diag_v2)",
    )
    ap.add_argument(
        "--phase",
        default="IN_PROGRESS",
        help="Phase filter for score rows",
    )
    ap.add_argument(
        "--min_rows",
        type=int,
        default=50,
        help="Minimum rows to report (skip if below)",
    )
    ap.add_argument(
        "--by_map",
        action="store_true",
        help="Report per (game_number, map_index)",
    )
    ap.add_argument(
        "--map_index",
        type=int,
        default=None,
        help="Filter score rows to this map_index",
    )
    ap.add_argument(
        "--game_number",
        type=int,
        default=None,
        help="Filter score rows to this game_number",
    )
    ap.add_argument(
        "--out_json",
        default="",
        help="If set, write overall + per_map metrics to this JSON file",
    )
    args = ap.parse_args()

    label_path = Path(args.label_input)
    score_path = Path(args.score_input)
    if not label_path.exists():
        print(f"Label file not found: {label_path}")
        return
    if not score_path.exists():
        print(f"Score file not found: {score_path}")
        return

    labels = _build_labels(label_path)
    phase = (args.phase or "IN_PROGRESS").strip()
    rows = _gather_rows(
        score_path,
        labels,
        phase,
        map_index=args.map_index,
        game_number=args.game_number,
    )

    if len(rows) < args.min_rows:
        print(f"Rows after join: {len(rows)} (min_rows={args.min_rows}). Skipping.")
        return

    pred = [r[0] for r in rows]
    y = [r[1] for r in rows]
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    auc = _compute_auc(y, pred)
    pred_inv = [1.0 - p for p in pred]
    auc_inv = _compute_auc(y, pred_inv)

    if auc is None or auc_inv is None:
        print("n_pos or n_neg is 0; cannot compute AUC.")
        return

    # Summary
    print(f"n_rows={len(rows)}  n_pos={n_pos}  n_neg={n_neg}")
    print(f"AUC(p_unshaped)     = {auc:.4f}")
    print(f"AUC(1-p_unshaped)  = {auc_inv:.4f}")
    if auc < 0.25 and auc_inv > 0.75:
        print("-> LIKELY INVERTED (p_unshaped predicts opposite of y)")
    elif 0.45 <= auc <= 0.55:
        print("-> NO SIGNAL / RANDOM")
    elif auc > 0.75:
        print("-> GOOD")
    else:
        print("-> WEAK / MIXED")

    out_data: dict[str, Any] = {
        "n_rows": len(rows),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc_p_unshaped": auc,
        "auc_1_minus_p_unshaped": auc_inv,
        "phase": phase,
        "label_input": str(label_path),
        "score_input": str(score_path),
    }

    if args.by_map:
        by_key: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
        for pu, yi, gn, mi, _ in rows:
            by_key[(gn, mi)].append((pu, yi))
        per_map: list[dict[str, Any]] = []
        for (gn, mi), pairs in by_key.items():
            if len(pairs) < args.min_rows:
                continue
            y_m = [p[1] for p in pairs]
            pred_m = [p[0] for p in pairs]
            auc_m = _compute_auc(y_m, pred_m)
            auc_inv_m = _compute_auc(y_m, [1.0 - p for p in pred_m])
            if auc_m is None:
                auc_m = float("nan")
            if auc_inv_m is None:
                auc_inv_m = float("nan")
            per_map.append({
                "game_number": gn,
                "map_index": mi,
                "n_rows": len(pairs),
                "n_pos": sum(y_m),
                "n_neg": len(y_m) - sum(y_m),
                "auc_p_unshaped": auc_m,
                "auc_1_minus_p_unshaped": auc_inv_m,
            })
        # Sort by AUC ascending (most inverted first)
        per_map.sort(key=lambda x: (x["auc_p_unshaped"], -x["n_rows"]))
        print("\nPer (game_number, map_index) — most inverted first:")
        for m in per_map:
            gn, mi = m["game_number"], m["map_index"]
            a = m["auc_p_unshaped"]
            ai = m["auc_1_minus_p_unshaped"]
            verdict = "LIKELY INVERTED" if (a < 0.25 and ai > 0.75) else ("NO SIGNAL" if 0.45 <= a <= 0.55 else ("GOOD" if a > 0.75 else "WEAK"))
            print(f"  game={gn} map={mi}  n={m['n_rows']}  AUC={a:.4f}  AUC_inv={ai:.4f}  -> {verdict}")
        out_data["per_map"] = per_map

    if args.out_json.strip():
        out_path = Path(args.out_json.strip())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
