"""
Config defaults and safe merge of partial updates.

Used by the backend when applying UI/control updates without overwriting
fields that were not sent.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from engine.models import Config


DEFAULTS: dict[str, Any] = {
    "source": "BO3",
    "match_id": None,
    "grid_series_id": None,
    "bo3_match_ids": None,
    "grid_series_ids": None,
    "primary_session_source": None,
    "primary_session_id": None,
    "bo3_auto_track": False,
    "bo3_auto_track_limit": 5,
    "bo3_auto_track_refresh_s": 30.0,
    "bo3_auto_track_probe_budget": 40,
    "grid_auto_track": False,
    "grid_auto_track_limit": 5,
    "grid_auto_track_refresh_s": 60.0,
    "poll_interval_s": 5.0,
    "contract_scope": "",
    "series_fmt": "",
    "prematch_series": None,
    "prematch_map": None,
    "prematch_locked": False,
    "lock_team_mapping": False,
    "market_delay_s": 0.0,
    "team_a_is_team_one": True,
    "replay_path": "logs/bo3_pulls.jsonl",
    "replay_loop": True,
    "replay_speed": 1.0,
    "replay_index": 0,
    "replay_contract_policy": "reject_point_like",
    "replay_point_transition_enabled": False,
    "replay_point_transition_sunset_epoch": None,
    "context_widening_enabled": False,
    "context_widen_beta": 0.25,
    "uncertainty_mult_min": 1.0,
    "uncertainty_mult_max": 1.35,
    "context_risk_weight_leverage": 0.4,
    "context_risk_weight_fragility": 0.4,
    "context_risk_weight_missingness": 0.2,
    "market_enabled": True,
    "kalshi_url": None,
    "kalshi_ticker": None,
    "market_delay_sec": 120,
    "market_poll_sec": 5,
    "market_side": None,
    "midround_v2_weight_profile": "current",
}


def _coerce_int_list(value: Any) -> Optional[list[int]]:
    """Coerce to list of int; single int/str -> [x]; None/empty -> None."""
    if value is None:
        return None
    if isinstance(value, list):
        out = []
        for v in value:
            if v is None:
                continue
            try:
                out.append(int(v) if not isinstance(v, int) else v)
            except (TypeError, ValueError):
                continue
        return out if out else None
    try:
        return [int(value)] if isinstance(value, str) and value.strip().isdigit() else [int(value)]
    except (TypeError, ValueError):
        return None


def _coerce_str_list(value: Any) -> Optional[list[str]]:
    """Coerce to list of non-empty str; single str -> [x]; None/empty -> None."""
    if value is None:
        return None
    if isinstance(value, list):
        out = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return out if out else None
    s = str(value).strip()
    return [s] if s else None


def _coerce_match_id(value: Any) -> Optional[int]:
    """Coerce match_id: None/'' -> None; int -> int; string of digits -> int. Else raise ValueError."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"invalid match_id: {value!r}")


def _coerce_finite_float(value: Any, fallback: float) -> float:
    """Coerce finite float, else fallback."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return f if math.isfinite(f) else fallback


def merge_config(current: Config, partial: dict[str, Any]) -> Config:
    """
    Return a new Config with only the keys present in `partial` updated.
    Unknown keys in `partial` are ignored. Does not mutate `current`.
    Normalizes source to uppercase and match_id to int or None.
    """
    allowed = set(DEFAULTS)
    updates = {k: v for k, v in partial.items() if k in allowed}
    if "grid_series_id" in updates and updates["grid_series_id"] is not None:
        updates["grid_series_id"] = str(updates["grid_series_id"]).strip() or None
    if getattr(current, "prematch_locked", False):
        updates.pop("prematch_series", None)
        updates.pop("prematch_map", None)
    if not updates:
        return current
    if "source" in updates and isinstance(updates["source"], str):
        updates["source"] = (updates["source"] or "BO3").strip().upper()
        if updates["source"] not in ("BO3", "GRID", "REPLAY"):
            updates["source"] = "BO3"
    if "match_id" in updates:
        updates["match_id"] = _coerce_match_id(updates["match_id"])
    if "bo3_match_ids" in updates:
        updates["bo3_match_ids"] = _coerce_int_list(updates["bo3_match_ids"])
    if "grid_series_ids" in updates:
        updates["grid_series_ids"] = _coerce_str_list(updates["grid_series_ids"])
    if "primary_session_source" in updates:
        v = updates["primary_session_source"]
        updates["primary_session_source"] = (str(v).strip().upper() if v is not None else "") or None
        if updates["primary_session_source"] and updates["primary_session_source"] not in ("BO3", "GRID"):
            updates["primary_session_source"] = None
    if "primary_session_id" in updates:
        v = updates["primary_session_id"]
        updates["primary_session_id"] = (str(v).strip() if v is not None else None) or None
    if "bo3_auto_track" in updates:
        updates["bo3_auto_track"] = bool(updates["bo3_auto_track"])
    if "bo3_auto_track_limit" in updates:
        v = updates["bo3_auto_track_limit"]
        try:
            n = int(v) if v is not None else 5
            updates["bo3_auto_track_limit"] = max(0, min(50, n))
        except (TypeError, ValueError):
            updates["bo3_auto_track_limit"] = 5
    if "bo3_auto_track_refresh_s" in updates:
        v = updates["bo3_auto_track_refresh_s"]
        try:
            f = float(v) if v is not None else 30.0
            updates["bo3_auto_track_refresh_s"] = max(10.0, f)
        except (TypeError, ValueError):
            updates["bo3_auto_track_refresh_s"] = 30.0
    if "bo3_auto_track_probe_budget" in updates:
        v = updates["bo3_auto_track_probe_budget"]
        try:
            n = int(v) if v is not None else 40
            updates["bo3_auto_track_probe_budget"] = max(5, min(200, n))
        except (TypeError, ValueError):
            updates["bo3_auto_track_probe_budget"] = 40
    if "grid_auto_track" in updates:
        updates["grid_auto_track"] = bool(updates["grid_auto_track"])
    if "grid_auto_track_limit" in updates:
        v = updates["grid_auto_track_limit"]
        try:
            n = int(v) if v is not None else 5
            updates["grid_auto_track_limit"] = max(0, min(50, n))
        except (TypeError, ValueError):
            updates["grid_auto_track_limit"] = 5
    if "grid_auto_track_refresh_s" in updates:
        v = updates["grid_auto_track_refresh_s"]
        try:
            f = float(v) if v is not None else 60.0
            updates["grid_auto_track_refresh_s"] = max(10.0, f)
        except (TypeError, ValueError):
            updates["grid_auto_track_refresh_s"] = 60.0
    if "midround_v2_weight_profile" in updates:
        v = updates["midround_v2_weight_profile"]
        if isinstance(v, str) and v.strip().lower() in ("learned_v1", "learned_v2", "learned_fit"):
            updates["midround_v2_weight_profile"] = v.strip().lower()
        else:
            updates["midround_v2_weight_profile"] = "current"
    if "replay_contract_policy" in updates:
        v = updates["replay_contract_policy"]
        updates["replay_contract_policy"] = str(v).strip().lower() if v is not None else "reject_point_like"
        if updates["replay_contract_policy"] != "reject_point_like":
            updates["replay_contract_policy"] = "reject_point_like"
    if "replay_point_transition_enabled" in updates:
        updates["replay_point_transition_enabled"] = bool(updates["replay_point_transition_enabled"])
    if "replay_point_transition_sunset_epoch" in updates:
        v = updates["replay_point_transition_sunset_epoch"]
        if v in (None, ""):
            updates["replay_point_transition_sunset_epoch"] = None
        else:
            try:
                updates["replay_point_transition_sunset_epoch"] = float(v)
            except (TypeError, ValueError):
                updates["replay_point_transition_sunset_epoch"] = None
    if "context_widening_enabled" in updates:
        updates["context_widening_enabled"] = bool(updates["context_widening_enabled"])
    if "context_widen_beta" in updates:
        default_beta = float(getattr(current, "context_widen_beta", 0.25))
        beta = _coerce_finite_float(updates["context_widen_beta"], default_beta)
        updates["context_widen_beta"] = max(0.0, min(2.0, beta))
    if "uncertainty_mult_min" in updates or "uncertainty_mult_max" in updates:
        default_min = max(1.0, _coerce_finite_float(getattr(current, "uncertainty_mult_min", 1.0), 1.0))
        default_max = max(default_min, _coerce_finite_float(getattr(current, "uncertainty_mult_max", 1.35), 1.35))
        umin = _coerce_finite_float(updates.get("uncertainty_mult_min"), default_min) if "uncertainty_mult_min" in updates else default_min
        umax = _coerce_finite_float(updates.get("uncertainty_mult_max"), default_max) if "uncertainty_mult_max" in updates else default_max
        umin = max(1.0, min(3.0, umin))
        umax = max(1.0, min(3.0, umax))
        if umax < umin:
            umax = umin
        updates["uncertainty_mult_min"] = umin
        updates["uncertainty_mult_max"] = umax
    if (
        "context_risk_weight_leverage" in updates
        or "context_risk_weight_fragility" in updates
        or "context_risk_weight_missingness" in updates
    ):
        w_lev = _coerce_finite_float(
            updates.get("context_risk_weight_leverage", getattr(current, "context_risk_weight_leverage", 0.4)),
            0.4,
        )
        w_frag = _coerce_finite_float(
            updates.get("context_risk_weight_fragility", getattr(current, "context_risk_weight_fragility", 0.4)),
            0.4,
        )
        w_miss = _coerce_finite_float(
            updates.get("context_risk_weight_missingness", getattr(current, "context_risk_weight_missingness", 0.2)),
            0.2,
        )
        w_lev = max(0.0, min(10.0, w_lev))
        w_frag = max(0.0, min(10.0, w_frag))
        w_miss = max(0.0, min(10.0, w_miss))
        total = w_lev + w_frag + w_miss
        if total <= 1e-9:
            w_lev, w_frag, w_miss = 0.4, 0.4, 0.2
            total = 1.0
        updates["context_risk_weight_leverage"] = w_lev / total
        updates["context_risk_weight_fragility"] = w_frag / total
        updates["context_risk_weight_missingness"] = w_miss / total
    d = {
        "source": getattr(current, "source"),
        "match_id": getattr(current, "match_id"),
        "grid_series_id": getattr(current, "grid_series_id", None),
        "bo3_match_ids": getattr(current, "bo3_match_ids", None),
        "grid_series_ids": getattr(current, "grid_series_ids", None),
        "primary_session_source": getattr(current, "primary_session_source", None),
        "primary_session_id": getattr(current, "primary_session_id", None),
        "bo3_auto_track": getattr(current, "bo3_auto_track", False),
        "bo3_auto_track_limit": getattr(current, "bo3_auto_track_limit", 5),
        "bo3_auto_track_refresh_s": getattr(current, "bo3_auto_track_refresh_s", 30.0),
        "bo3_auto_track_probe_budget": getattr(current, "bo3_auto_track_probe_budget", 40),
        "grid_auto_track": getattr(current, "grid_auto_track", False),
        "grid_auto_track_limit": getattr(current, "grid_auto_track_limit", 5),
        "grid_auto_track_refresh_s": getattr(current, "grid_auto_track_refresh_s", 60.0),
        "poll_interval_s": getattr(current, "poll_interval_s"),
        "contract_scope": getattr(current, "contract_scope"),
        "series_fmt": getattr(current, "series_fmt"),
        "prematch_series": getattr(current, "prematch_series"),
        "prematch_map": getattr(current, "prematch_map"),
        "prematch_locked": getattr(current, "prematch_locked", False),
        "lock_team_mapping": getattr(current, "lock_team_mapping"),
        "market_delay_s": getattr(current, "market_delay_s"),
        "team_a_is_team_one": getattr(current, "team_a_is_team_one", True),
        "replay_path": getattr(current, "replay_path", None),
        "replay_loop": getattr(current, "replay_loop", True),
        "replay_speed": getattr(current, "replay_speed", 1.0),
        "replay_index": getattr(current, "replay_index", 0),
        "replay_contract_policy": getattr(current, "replay_contract_policy", "reject_point_like"),
        "replay_point_transition_enabled": getattr(current, "replay_point_transition_enabled", False),
        "replay_point_transition_sunset_epoch": getattr(current, "replay_point_transition_sunset_epoch", None),
        "context_widening_enabled": getattr(current, "context_widening_enabled", False),
        "context_widen_beta": getattr(current, "context_widen_beta", 0.25),
        "uncertainty_mult_min": getattr(current, "uncertainty_mult_min", 1.0),
        "uncertainty_mult_max": getattr(current, "uncertainty_mult_max", 1.35),
        "context_risk_weight_leverage": getattr(current, "context_risk_weight_leverage", 0.4),
        "context_risk_weight_fragility": getattr(current, "context_risk_weight_fragility", 0.4),
        "context_risk_weight_missingness": getattr(current, "context_risk_weight_missingness", 0.2),
        "market_enabled": getattr(current, "market_enabled", True),
        "kalshi_url": getattr(current, "kalshi_url", None),
        "kalshi_ticker": getattr(current, "kalshi_ticker", None),
        "market_delay_sec": getattr(current, "market_delay_sec", 120),
        "market_poll_sec": getattr(current, "market_poll_sec", 5),
        "market_side": getattr(current, "market_side", None),
        "midround_v2_weight_profile": getattr(current, "midround_v2_weight_profile", "current"),
    }
    d.update(updates)
    # Enforce minimum 5s BO3 poll interval
    pi = d.get("poll_interval_s", 5.0)
    try:
        d["poll_interval_s"] = max(5.0, float(pi))
    except (TypeError, ValueError):
        d["poll_interval_s"] = 5.0
    return Config(**d)
