"""Log persistence, schema migration, in-play persistence, and metrics."""
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from .paths import (
    LOG_PATH, LOG_COLUMNS,
    INPLAY_LOG_PATH, INPLAY_LOG_COLUMNS,
    INPLAY_RESULTS_PATH, INPLAY_RESULTS_COLUMNS,
    INPLAY_MAP_RESULTS_PATH, INPLAY_MAP_RESULTS_COLUMNS,
    CS2_REPLAY_SNAPSHOT_PARQUET_PATH,
    CS2_ML_FEATURE_PARQUET_PATH,
    CS2_REPLAY_SNAPSHOT_COLUMNS,
    CS2_ML_FEATURE_COLUMNS,
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


# --- CS2 replay snapshot + ML feature parquet persistence ---
CS2_FEATURE_SCHEMA_VERSION = 1
BO3_LIVE_CAPTURE_SCHEMA_VERSION = "bo3_live_capture_contract.v1"


def _pick_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_cs2_side(value: Any) -> Optional[str]:
    text = _str_or_none(value)
    if text is None:
        return None
    upper = text.upper()
    if upper in ("T", "TERRORIST"):
        return "T"
    if upper in ("CT", "COUNTER-TERRORIST", "COUNTER TERRORIST"):
        return "CT"
    if upper == "UNKNOWN":
        return "Unknown"
    return text


def build_bo3_live_capture_context(feed: dict) -> dict:
    payload = feed.get("payload") if isinstance(feed, dict) else None
    if not isinstance(payload, dict):
        payload = {}
    team_a_is_team_one = bool(feed.get("team_a_is_team_one", True)) if isinstance(feed, dict) else True
    team_one = payload.get("team_one") if isinstance(payload.get("team_one"), dict) else {}
    team_two = payload.get("team_two") if isinstance(payload.get("team_two"), dict) else {}
    team_a = team_one if team_a_is_team_one else team_two
    team_b = team_two if team_a_is_team_one else team_one
    return {
        "team_a_is_team_one": team_a_is_team_one,
        "raw_ts_utc": _str_or_none(feed.get("raw_ts_utc")) if isinstance(feed, dict) else None,
        "raw_provider_event_id": _str_or_none(payload.get("provider_event_id")),
        "raw_seq_index": _int_or_none(payload.get("seq_index")),
        "raw_sent_time": _str_or_none(payload.get("sent_time")),
        "raw_updated_at": _str_or_none(payload.get("updated_at")),
        "raw_record_path": _str_or_none(feed.get("raw_record_path")) if isinstance(feed, dict) else None,
        "game_number": _int_or_none(payload.get("game_number")),
        "round_number": _int_or_none(payload.get("round_number")),
        "round_phase": _str_or_none(payload.get("round_phase") or payload.get("phase")),
        "team_a_id": _int_or_none(team_a.get("id")),
        "team_b_id": _int_or_none(team_b.get("id")),
        "team_a_provider_id": _str_or_none(team_a.get("provider_id")),
        "team_b_provider_id": _str_or_none(team_b.get("provider_id")),
        "team_a_side_used": _normalize_cs2_side(team_a.get("side")),
        "team_b_side_used": _normalize_cs2_side(team_b.get("side")),
    }


def augment_bo3_live_capture_contract(snapshot_row: dict, feed: dict, live_frame: Optional[dict] = None) -> dict:
    row = dict(snapshot_row or {})
    frame = dict(live_frame or {})
    ctx = build_bo3_live_capture_context(feed if isinstance(feed, dict) else {})
    row.update({
        "schema_version": BO3_LIVE_CAPTURE_SCHEMA_VERSION,
        "live_source": "BO3",
        "capture_ts_iso": _pick_present(row.get("snapshot_ts_iso"), row.get("snapshot_ts")),
        "team_a_is_team_one": ctx.get("team_a_is_team_one"),
        "raw_ts_utc": ctx.get("raw_ts_utc"),
        "raw_provider_event_id": ctx.get("raw_provider_event_id"),
        "raw_seq_index": ctx.get("raw_seq_index"),
        "raw_sent_time": ctx.get("raw_sent_time"),
        "raw_updated_at": ctx.get("raw_updated_at"),
        "raw_record_path": ctx.get("raw_record_path"),
        "game_number": _pick_present(frame.get("game_number"), ctx.get("game_number")),
        "round_number": _pick_present(frame.get("round_number"), ctx.get("round_number")),
        "round_phase": _pick_present(frame.get("round_phase"), ctx.get("round_phase")),
        "a_side": _pick_present(frame.get("a_side"), ctx.get("team_a_side_used"), row.get("drv_team_a_side")),
        "team_a_id": ctx.get("team_a_id"),
        "team_b_id": ctx.get("team_b_id"),
        "team_a_provider_id": ctx.get("team_a_provider_id"),
        "team_b_provider_id": ctx.get("team_b_provider_id"),
        "team_a_side_used": _pick_present(ctx.get("team_a_side_used"), frame.get("team_a_side_used"), row.get("drv_team_a_side")),
        "team_b_side_used": _pick_present(ctx.get("team_b_side_used"), frame.get("team_b_side_used")),
        "bomb_planted": _pick_present(frame.get("bomb_planted"), row.get("drv_bomb_planted"), row.get("mid_bomb_planted")),
        "round_time_remaining_s": _pick_present(frame.get("round_time_remaining_s"), row.get("drv_round_time_remaining_s"), row.get("mid_time_remaining_s")),
        "alive_count_a": _pick_present(frame.get("alive_count_a"), row.get("drv_alive_count_a"), row.get("team_a_alive_count")),
        "alive_count_b": _pick_present(frame.get("alive_count_b"), row.get("drv_alive_count_b"), row.get("team_b_alive_count")),
        "hp_alive_total_a": _pick_present(frame.get("hp_alive_total_a"), row.get("drv_hp_alive_total_a"), row.get("team_a_hp_alive_total")),
        "hp_alive_total_b": _pick_present(frame.get("hp_alive_total_b"), row.get("drv_hp_alive_total_b"), row.get("team_b_hp_alive_total")),
        "cash_total_a": _pick_present(frame.get("cash_total_a"), row.get("team_a_cash_total"), row.get("team_a_money_total")),
        "cash_total_b": _pick_present(frame.get("cash_total_b"), row.get("team_b_cash_total"), row.get("team_b_money_total")),
        "loadout_est_total_a": _pick_present(frame.get("loadout_est_total_a"), row.get("team_a_loadout_est_total")),
        "loadout_est_total_b": _pick_present(frame.get("loadout_est_total_b"), row.get("team_b_loadout_est_total")),
        "alive_loadout_total_a": _pick_present(frame.get("alive_loadout_total_a"), row.get("drv_alive_loadout_est_total_a"), row.get("team_a_alive_loadout_est_total")),
        "alive_loadout_total_b": _pick_present(frame.get("alive_loadout_total_b"), row.get("drv_alive_loadout_est_total_b"), row.get("team_b_alive_loadout_est_total")),
        "armor_alive_total_a": _pick_present(frame.get("armor_alive_total_a"), row.get("armor_alive_total_a")),
        "armor_alive_total_b": _pick_present(frame.get("armor_alive_total_b"), row.get("armor_alive_total_b")),
        "intraround_state_source": _pick_present(frame.get("intraround_state_source"), row.get("intraround_state_source")),
        "rail_low": _pick_present(row.get("rail_low"), row.get("rail_p_if_next_round_loss")),
        "rail_high": _pick_present(row.get("rail_high"), row.get("rail_p_if_next_round_win")),
    })
    return row


def should_persist_bo3_live_capture_contract(
    snapshot_row: dict,
    *,
    bo3_source_mode: Optional[str] = None,
    snapshot_status: Optional[str] = None,
) -> bool:
    if snapshot_row.get("live_source") != "BO3":
        return False
    if bo3_source_mode and bo3_source_mode != "LIVE (poller feed file)":
        return False
    if snapshot_status and snapshot_status != "live":
        return False
    if not _str_or_none(snapshot_row.get("match_id")):
        return False
    if not _str_or_none(snapshot_row.get("raw_record_path")):
        return False
    has_raw_identity = bool(
        _str_or_none(snapshot_row.get("raw_provider_event_id"))
        or snapshot_row.get("raw_seq_index") is not None
    )
    return has_raw_identity


def _parquet_append_row(path: Path, columns: list, entry: dict) -> None:
    """Append one row to a parquet file. Read-concat-write for correctness. Preserves column order and nulls."""
    row = {k: entry.get(k) for k in columns}
    df_new = pd.DataFrame([row], columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            df_existing = pd.read_parquet(path)
            existing_cols = [c for c in columns if c in df_existing.columns]
            missing_in_file = [c for c in columns if c not in df_existing.columns]
            if missing_in_file:
                for c in missing_in_file:
                    df_existing[c] = None
            df_existing = df_existing[columns]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
    df_combined.to_parquet(path, index=False)


def persist_cs2_replay_snapshot(entry: dict) -> None:
    """Persist one CS2 replay snapshot row to parquet. Filters to allowed columns, preserves order and nulls."""
    _parquet_append_row(CS2_REPLAY_SNAPSHOT_PARQUET_PATH, CS2_REPLAY_SNAPSHOT_COLUMNS, entry)


def persist_cs2_ml_feature_snapshot(entry: dict) -> None:
    """Persist one CS2 ML feature row to parquet. Filters to allowed columns, preserves order and nulls."""
    _parquet_append_row(CS2_ML_FEATURE_PARQUET_PATH, CS2_ML_FEATURE_COLUMNS, entry)


def _f(x: Any) -> Optional[float]:
    """Coerce to float or None for derived ML features."""
    if x is None:
        return None
    try:
        v = float(x)
        return v if (v == v) else None
    except (TypeError, ValueError):
        return None


def derive_cs2_ml_feature_row(snapshot_row: dict) -> dict:
    """Build a curated ML feature row from the full replay snapshot_row. Adds derived numeric features."""
    mid = str(snapshot_row.get("match_id", "")).strip() if snapshot_row.get("match_id") else None
    if not mid:
        mid = None
    p_hat = _f(snapshot_row.get("p_hat"))
    p_hat_map = _f(snapshot_row.get("p_hat_map"))
    market_mid = _f(snapshot_row.get("market_mid"))
    band_lo = _f(snapshot_row.get("band_lo"))
    band_hi = _f(snapshot_row.get("band_hi"))
    band_state_lo = _f(snapshot_row.get("band_state_lo"))
    band_state_hi = _f(snapshot_row.get("band_state_hi"))
    state_lo = _f(snapshot_row.get("state_bound_lower"))
    state_hi = _f(snapshot_row.get("state_bound_upper"))

    round_band_width_pp = (band_state_hi - band_state_lo) * 100.0 if band_state_lo is not None and band_state_hi is not None else None
    map_band_width_pp = (state_hi - state_lo) * 100.0 if state_lo is not None and state_hi is not None else None
    kappa_band_width_pp = (band_hi - band_lo) * 100.0 if band_lo is not None and band_hi is not None else None

    dist_mid_to_round_lo_pp = (market_mid - band_state_lo) * 100.0 if market_mid is not None and band_state_lo is not None else None
    dist_mid_to_round_hi_pp = (band_state_hi - market_mid) * 100.0 if market_mid is not None and band_state_hi is not None else None
    dist_mid_to_map_lo_pp = (market_mid - state_lo) * 100.0 if market_mid is not None and state_lo is not None else None
    dist_mid_to_map_hi_pp = (state_hi - market_mid) * 100.0 if market_mid is not None and state_hi is not None else None
    dist_mid_to_kappa_lo_pp = (market_mid - band_lo) * 100.0 if market_mid is not None and band_lo is not None else None
    dist_mid_to_kappa_hi_pp = (band_hi - market_mid) * 100.0 if market_mid is not None and band_hi is not None else None

    round_w = (band_state_hi - band_state_lo) if (band_state_lo is not None and band_state_hi is not None and band_state_hi > band_state_lo) else None
    p_hat_pos_in_round_band = (p_hat - band_state_lo) / round_w if (round_w and round_w > 1e-9 and p_hat is not None and band_state_lo is not None) else None
    map_w = (state_hi - state_lo) if (state_lo is not None and state_hi is not None and state_hi > state_lo) else None
    p_hat_pos_in_map_band = (p_hat - state_lo) / map_w if (map_w and map_w > 1e-9 and p_hat is not None and state_lo is not None) else None

    q = _f(snapshot_row.get("q_intra_round_win_a"))
    q_minus_market_mid_pp = (q - market_mid) * 100.0 if q is not None and market_mid is not None else None
    p_hat_minus_p_hat_map_pp = (p_hat - p_hat_map) * 100.0 if p_hat is not None and p_hat_map is not None else None

    round_importance_proxy = None
    if state_lo is not None and state_hi is not None and market_mid is not None:
        try:
            mid_to_lo = abs(market_mid - state_lo)
            mid_to_hi = abs(market_mid - state_hi)
            span = (state_hi - state_lo) or 1e-9
            round_importance_proxy = min(mid_to_lo, mid_to_hi) / span
        except Exception:
            pass

    return {
        "snapshot_ts_iso": snapshot_row.get("snapshot_ts_iso"),
        "snapshot_ts_epoch_ms": snapshot_row.get("snapshot_ts_epoch_ms"),
        "feature_schema_version": CS2_FEATURE_SCHEMA_VERSION,
        "match_id": mid,
        "series_fmt": snapshot_row.get("series_fmt"),
        "contract_scope": snapshot_row.get("contract_scope"),
        "maps_a_won": snapshot_row.get("maps_a_won"),
        "maps_b_won": snapshot_row.get("maps_b_won"),
        "rounds_a": snapshot_row.get("rounds_a"),
        "rounds_b": snapshot_row.get("rounds_b"),
        "map_index": snapshot_row.get("map_index"),
        "total_rounds": snapshot_row.get("total_rounds"),
        "market_bid": snapshot_row.get("market_bid"),
        "market_ask": snapshot_row.get("market_ask"),
        "market_mid": market_mid,
        "p_hat": p_hat,
        "p_hat_map": p_hat_map,
        "p0_map": snapshot_row.get("p0_map"),
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_state_lo": band_state_lo,
        "band_state_hi": band_state_hi,
        "state_bound_lower": state_lo,
        "state_bound_upper": state_hi,
        "band_lo_map": snapshot_row.get("band_lo_map"),
        "band_hi_map": snapshot_row.get("band_hi_map"),
        "q_intra_round_win_a": q,
        "q_intra_round_win_a_source": snapshot_row.get("q_intra_round_win_a_source"),
        "q_intra_round_win_a_reason": snapshot_row.get("q_intra_round_win_a_reason"),
        "branch_endpoint_source_mode": snapshot_row.get("branch_endpoint_source_mode"),
        "branch_endpoint_source_used": snapshot_row.get("branch_endpoint_source_used"),
        "branch_endpoint_source_reason": snapshot_row.get("branch_endpoint_source_reason"),
        "drv_valid_microstate": snapshot_row.get("drv_valid_microstate"),
        "drv_valid_roundstate": snapshot_row.get("drv_valid_roundstate"),
        "drv_team_a_side": snapshot_row.get("drv_team_a_side"),
        "drv_bomb_planted": snapshot_row.get("drv_bomb_planted"),
        "drv_round_phase": snapshot_row.get("drv_round_phase"),
        "drv_round_time_remaining_s": snapshot_row.get("drv_round_time_remaining_s"),
        "drv_alive_count_a": snapshot_row.get("drv_alive_count_a"),
        "drv_alive_count_b": snapshot_row.get("drv_alive_count_b"),
        "drv_hp_alive_total_a": snapshot_row.get("drv_hp_alive_total_a"),
        "drv_hp_alive_total_b": snapshot_row.get("drv_hp_alive_total_b"),
        "drv_alive_loadout_est_total_a": snapshot_row.get("drv_alive_loadout_est_total_a"),
        "drv_alive_loadout_est_total_b": snapshot_row.get("drv_alive_loadout_est_total_b"),
        "drv_alive_cash_total_a": snapshot_row.get("drv_alive_cash_total_a"),
        "drv_alive_cash_total_b": snapshot_row.get("drv_alive_cash_total_b"),
        "team_a_money_total": snapshot_row.get("team_a_money_total"),
        "team_b_money_total": snapshot_row.get("team_b_money_total"),
        "team_a_loadout_est_total": snapshot_row.get("team_a_loadout_est_total"),
        "team_b_loadout_est_total": snapshot_row.get("team_b_loadout_est_total"),
        "econ_latched_a": snapshot_row.get("econ_latched_a"),
        "econ_latched_b": snapshot_row.get("econ_latched_b"),
        "round_transition_tick_detected": snapshot_row.get("round_transition_tick_detected"),
        "round_transition_latched_new_round_context": snapshot_row.get("round_transition_latched_new_round_context"),
        "round_snap_applied_this_tick": snapshot_row.get("round_snap_applied_this_tick"),
        "live_source_selected": snapshot_row.get("live_source_selected"),
        "grid_valid": snapshot_row.get("grid_valid"),
        "grid_completeness_score": snapshot_row.get("grid_completeness_score"),
        "round_band_width_pp": round_band_width_pp,
        "map_band_width_pp": map_band_width_pp,
        "kappa_band_width_pp": kappa_band_width_pp,
        "dist_mid_to_round_lo_pp": dist_mid_to_round_lo_pp,
        "dist_mid_to_round_hi_pp": dist_mid_to_round_hi_pp,
        "dist_mid_to_map_lo_pp": dist_mid_to_map_lo_pp,
        "dist_mid_to_map_hi_pp": dist_mid_to_map_hi_pp,
        "dist_mid_to_kappa_lo_pp": dist_mid_to_kappa_lo_pp,
        "dist_mid_to_kappa_hi_pp": dist_mid_to_kappa_hi_pp,
        "p_hat_pos_in_round_band": p_hat_pos_in_round_band,
        "p_hat_pos_in_map_band": p_hat_pos_in_map_band,
        "q_minus_market_mid_pp": q_minus_market_mid_pp,
        "p_hat_minus_p_hat_map_pp": p_hat_minus_p_hat_map_pp,
        "round_importance_proxy": round_importance_proxy,
    }


def show_inplay_log_paths():
    st.caption(f"In-play logs: {INPLAY_LOG_PATH}")
    st.caption(f"Match results: {INPLAY_RESULTS_PATH}")
    st.caption(f"Map results: {INPLAY_MAP_RESULTS_PATH}")

