"""Log persistence, schema migration, in-play persistence, and metrics."""
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from .paths import (
    LOG_PATH, LOG_COLUMNS,
    INPLAY_LOG_PATH, INPLAY_LOG_COLUMNS,
    INPLAY_RESULTS_PATH, INPLAY_RESULTS_COLUMNS,
    INPLAY_MAP_RESULTS_PATH, INPLAY_MAP_RESULTS_COLUMNS,
)
from .odds import implied_prob_from_american
from .data import sniff_bad_csv, read_csv_tolerant


def migrate_log_schema(path: Path, out_path: Optional[Path] = None) -> int:
    if out_path is None:
        out_path = path
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        file_header = next(rdr)
        old_len = len(file_header)
        new_len = len(LOG_COLUMNS)
        for row in rdr:
            if len(row) == new_len:
                d = dict(zip(LOG_COLUMNS, row))
            elif len(row) == old_len:
                d = dict(zip(file_header, row))
            else:
                continue
            rows.append({k: d.get(k, "") for k in LOG_COLUMNS})
    seen = set()
    deduped = []
    for d in rows:
        key = (d.get("timestamp", ""), d.get("game", ""), d.get("team_a", ""), d.get("team_b", ""), d.get("adj_gap", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
    try:
        deduped.sort(key=lambda r: r.get("timestamp", ""))
    except Exception:
        pass
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        w.writerows(deduped)
    return len(deduped)


def recompute_metrics_from_logs(rows: list) -> dict:
    metrics = {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0}
    for r in rows:
        metrics["total"] += 1
        mkt_type = str(r.get("market_type", "")).upper()
        if mkt_type == "3WAY":
            try:
                sel_ev = float(r.get("selected_ev_pct", ""))
                if not (sel_ev == sel_ev):
                    metrics["no_bet"] += 1
                    continue
            except Exception:
                metrics["no_bet"] += 1
                continue
            try:
                oa = float(r.get("odds_a20", 0) or 0)
                od = float(r.get("odds_draw", 0) or 0)
                ob = float(r.get("odds_b02", 0) or 0)
                inv_map = {"A2-0": (1.0 / oa) if oa > 1.0 else 0.0,
                           "DRAW": (1.0 / od) if od > 1.0 else 0.0,
                           "B0-2": (1.0 / ob) if ob > 1.0 else 0.0}
                vals = [x for x in [oa, od, ob] if x and x > 1.0]
                s = sum(1.0 / x for x in vals) if vals else 1.0
                imp = {k: (v / s if s > 0 else 0.0) for k, v in inv_map.items()}
                fav_outcome = max(imp.items(), key=lambda kv: kv[1])[0] if imp else "A2-0"
                sel = str(r.get("selected_outcome", ""))
                if sel_ev <= 0:
                    metrics["no_bet"] += 1
                else:
                    if sel == fav_outcome:
                        metrics["fav_value"] += 1
                    else:
                        metrics["dog_value"] += 1
            except Exception:
                metrics["no_bet"] += 1
            continue
        try:
            odds_a = int(float(r.get("odds_a", 0)))
            odds_b = int(float(r.get("odds_b", 0)))
            ev_a = float(r.get("ev_a_dec", r.get("ev_a_pct", -1e9)))
            ev_b = float(r.get("ev_b_dec", r.get("ev_b_pct", -1e9)))
        except Exception:
            metrics["no_bet"] += 1
            continue
        imp_a = implied_prob_from_american(odds_a) if odds_a else 0.0
        imp_b = implied_prob_from_american(odds_b) if odds_b else 0.0
        market_fav = "A" if imp_a >= imp_b else "B"
        if ev_a <= 0 and ev_b <= 0:
            metrics["no_bet"] += 1
        else:
            pick = "A" if ev_a >= ev_b else "B"
            if pick == market_fav:
                metrics["fav_value"] += 1
            else:
                metrics["dog_value"] += 1
    return metrics


def load_persisted_logs() -> list:
    if not LOG_PATH.exists():
        return []
    try:
        df = pd.read_csv(LOG_PATH)
    except pd.errors.ParserError as e:
        st.error(f"Failed to read log CSV ({LOG_PATH.name}): {e}")
        header, exp, bad = sniff_bad_csv(LOG_PATH)
        if bad:
            st.warning(f"Header has {exp} columns. Problem rows: {bad}")
        df = read_csv_tolerant(LOG_PATH)
        st.warning(f"Loaded log after skipping malformed rows. Rows kept: {len(df)}")
    except Exception as e:
        st.error(f"Error reading log CSV: {e}")
        return []
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[LOG_COLUMNS].to_dict(orient="records")


def persist_log_row(entry: dict):
    row = {k: entry.get(k, "") for k in LOG_COLUMNS}
    df = pd.DataFrame([row], columns=LOG_COLUMNS)
    write_header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode="a", header=write_header, index=False, line_terminator="\n")


def init_metrics():
    if "logs" not in st.session_state:
        st.session_state["logs"] = load_persisted_logs()
    st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])


def update_metrics_binary(ev_a: float, ev_b: float, odds_a: int, odds_b: int):
    m = st.session_state.get("metrics", {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0})
    m["total"] += 1
    imp_a = implied_prob_from_american(odds_a) if odds_a else 0.0
    imp_b = implied_prob_from_american(odds_b) if odds_b else 0.0
    market_fav = "A" if imp_a >= imp_b else "B"
    if (ev_a is not None and ev_a > 0) or (ev_b is not None and ev_b > 0):
        pick = "A" if (ev_a or -1e9) >= (ev_b or -1e9) else "B"
        best_ev = ev_a if pick == "A" else ev_b
        if best_ev is None or best_ev <= 0:
            m["no_bet"] += 1
        else:
            if pick == market_fav:
                m["fav_value"] += 1
            else:
                m["dog_value"] += 1
    else:
        m["no_bet"] += 1
    st.session_state["metrics"] = m


def update_metrics_3way(
    selected_ev_pct: Optional[float], selected_outcome: Optional[str],
    odds_a20: float, odds_draw: float, odds_b02: float
):
    m = st.session_state.get("metrics", {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0})
    m["total"] += 1
    if not selected_outcome or selected_ev_pct is None or selected_ev_pct <= 0:
        m["no_bet"] += 1
    else:
        oa, od, ob = odds_a20, odds_draw, odds_b02
        inv_map = {"A2-0": (1.0 / oa) if oa and oa > 1 else 0.0,
                   "DRAW": (1.0 / od) if od and od > 1 else 0.0,
                   "B0-2": (1.0 / ob) if ob and ob > 1 else 0.0}
        s = sum(inv_map.values()) or 1.0
        imp = {k: v / s for k, v in inv_map.items()}
        fav_outcome = max(imp.items(), key=lambda kv: kv[1])[0]
        if selected_outcome == fav_outcome:
            m["fav_value"] += 1
        else:
            m["dog_value"] += 1
    st.session_state["metrics"] = m


def log_row(entry: dict):
    st.session_state["logs"].append({k: entry.get(k, "") for k in LOG_COLUMNS})
    try:
        persist_log_row(entry)
    except Exception as e:
        st.warning(f"Could not persist log row: {e}")
    st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])


def export_logs_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get("logs", []), columns=LOG_COLUMNS)


# --- In-play log persistence ---
def _ensure_csv(path: Path, columns: list):
    """Ensure CSV exists and has the requested schema."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([], columns=columns).to_csv(path, index=False)
        return
    try:
        existing_cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        raise
    missing = [c for c in columns if c not in existing_cols]
    if not missing and existing_cols == columns:
        return
    df = pd.read_csv(path)
    for c in missing:
        df[c] = ""
    extra = [c for c in df.columns if c not in columns]
    df = df[columns + extra]
    df.to_csv(path, index=False)


def persist_inplay_row(entry: dict):
    _ensure_csv(INPLAY_LOG_PATH, INPLAY_LOG_COLUMNS)
    row = {k: entry.get(k, "") for k in INPLAY_LOG_COLUMNS}
    pd.DataFrame([row], columns=INPLAY_LOG_COLUMNS).to_csv(
        INPLAY_LOG_PATH, mode="a", header=False, index=False
    )


def persist_inplay_result(match_id: str, game: str, team_a: str, team_b: str, winner: str):
    _ensure_csv(INPLAY_RESULTS_PATH, INPLAY_RESULTS_COLUMNS)
    row = {
        "match_id": match_id, "game": game, "team_a": team_a, "team_b": team_b,
        "winner": winner, "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    pd.DataFrame([row], columns=INPLAY_RESULTS_COLUMNS).to_csv(
        INPLAY_RESULTS_PATH, mode="a", header=False, index=False
    )


def persist_inplay_map_result(
    match_id: str, game: str, map_index: int, map_name: str,
    team_a: str, team_b: str, winner: str
):
    _ensure_csv(INPLAY_MAP_RESULTS_PATH, INPLAY_MAP_RESULTS_COLUMNS)
    row = {
        "match_id": match_id, "game": game, "map_index": int(map_index), "map_name": str(map_name or ""),
        "team_a": team_a, "team_b": team_b, "winner": winner,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    pd.DataFrame([row], columns=INPLAY_MAP_RESULTS_COLUMNS).to_csv(
        INPLAY_MAP_RESULTS_PATH, mode="a", header=False, index=False
    )


def show_inplay_log_paths():
    st.caption(f"In-play logs: {INPLAY_LOG_PATH}")
    st.caption(f"Match results: {INPLAY_RESULTS_PATH}")
    st.caption(f"Map results: {INPLAY_MAP_RESULTS_PATH}")
