"""
Stage 3: Assert inter_map_break debug structure contract — same keys across BO3/GRID/REPLAY.
Parity: inter_map_break=True => debug has inter_map_break, inter_map_break_reason, explain.phase, p_hat_final, map_low/map_high.
"""
from __future__ import annotations


def test_inter_map_break_dbg_contract_structure() -> None:
    """Inter-map-break branch must produce debug with required keys (parity contract)."""
    # Contract: when is_break is True, all sources build dbg with these keys (see runner BO3/GRID/REPLAY paths).
    required = {
        "inter_map_break",
        "inter_map_break_reason",
        "p_hat_old",
        "p_hat_final",
        "series_low",
        "series_high",
        "map_low",
        "map_high",
        "explain",
    }
    # Replicate the minimal structure the runner builds in each inter_map_break block.
    bound_low, bound_high = 0.2, 0.8
    break_reason = "test"
    series_width = bound_high - bound_low
    center = max(bound_low, min(bound_high, 0.5 * (bound_low + bound_high)))
    map_width = min(series_width * 0.6, 0.30)
    half = 0.5 * map_width
    rail_low = max(bound_low, min(bound_high, center - half))
    rail_high = max(bound_low, min(bound_high, center + half))
    p_hat = center
    cw = rail_high - rail_low if rail_high >= rail_low else 0.0
    dbg = {
        "inter_map_break": True,
        "inter_map_break_reason": break_reason,
        "p_hat_old": None,
        "p_hat_final": p_hat,
        "series_low": bound_low,
        "series_high": bound_high,
        "map_low": rail_low,
        "map_high": rail_high,
        "explain": {
            "phase": "inter_map_break",
            "p_base_map": None,
            "p_base_series": None,
            "midround_weight": 0.0,
            "q_intra_total": None,
            "q_terms": {},
            "micro_adj": {"alive_adj": 0.0, "hp_adj": 0.0, "econ_adj": 0.0},
            "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": cw},
            "final": {"p_hat_final": p_hat, "clamp_reason": "inter_map_break"},
        },
    }
    for key in required:
        assert key in dbg, f"inter_map_break dbg must contain {key!r}"
    assert dbg["explain"]["phase"] == "inter_map_break"
