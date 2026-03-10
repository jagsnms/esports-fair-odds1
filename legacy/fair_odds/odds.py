"""Odds conversion, fair odds mapping, and decision helpers."""

import math


def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1


def decimal_to_american(o: float) -> int:
    try:
        o = float(o)
    except Exception:
        return 0
    if o <= 1.0:
        return 0
    return int(round((o - 1.0) * 100)) if o >= 2.0 else int(round(-100.0 / (o - 1.0)))


def implied_prob_from_american(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def calculate_fair_odds_curve(
    gap: float, base_C=30, alpha=0.03, tail_cutoff=14, L=0.97, k=0.09, x0=0, v=1.0
):
    """Original hand-tuned mapping from gap -> p."""
    if abs(gap) <= tail_cutoff:
        C_dynamic = base_C / (1 + alpha * abs(gap))
        p_a = 1 / (1 + 10 ** (-gap / C_dynamic))
    else:
        p_a = L / ((1 + math.exp(-k * (gap - x0))) ** v)
        p_b = L - p_a
        total = p_a + p_b
        p_a /= total
    return p_a


def calculate_fair_odds_from_p(p_a: float):
    p_b = 1 - p_a
    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)
    return fair_a, fair_b


def logistic_mapping(adj_gap: float, a: float, b: float) -> float:
    """p = 1 / (1 + exp(-(a + b*gap)))"""
    return 1.0 / (1.0 + math.exp(-(a + b * adj_gap)))


def color_ev(ev: float) -> str:
    if ev >= 40:
        return f"✅ **{round(ev, 2)}%**"
    elif 11 <= ev < 40:
        return f"🟡 {round(ev, 2)}%"
    else:
        return f"🔴 {round(ev, 2)}%"


def prob_to_fair_american(p: float) -> int:
    """Convert a probability to no-vig fair odds (American)."""
    p = max(1e-9, min(1 - 1e-9, float(p)))
    dec = 1.0 / p
    return decimal_to_american(dec)


def compute_bo2_probs(p_map: float, k: float = 1.0):
    """
    From per-map win prob p_map (Team A) produce series-level probs for a BO2:
      A2-0, DRAW(1-1), B0-2.
    """
    p = max(0.0, min(1.0, float(p_map)))
    p_a20 = p * p
    p_draw = 2 * p * (1 - p)
    p_b02 = (1 - p) * (1 - p)
    p_draw *= max(0.0, float(k))
    s = p_a20 + p_draw + p_b02
    if s <= 0:
        return {"A2-0": 0.0, "DRAW": 0.0, "B0-2": 0.0}
    return {"A2-0": p_a20 / s, "DRAW": p_draw / s, "B0-2": p_b02 / s}


def ev_pct_decimal(prob: float, dec_odds: float) -> float:
    """EV% for a discrete outcome with probability 'prob' at DECIMAL odds."""
    if dec_odds is None or dec_odds <= 1.0:
        return float("nan")
    return (prob * dec_odds - 1.0) * 100.0


def decide_bet(
    p_model: float,
    odds_a: int,
    odds_b: int,
    n_matches_a: int,
    n_matches_b: int,
    min_edge_pct: float,
    prob_gap_pp: float,
    shrink_target: int,
):
    """
    Returns dict: p_decide, ev_a_dec, ev_b_dec, choice ('A'/'B'/None), reason (str).
    """
    eff_matches = min(n_matches_a, n_matches_b)
    lam = min(1.0, eff_matches / float(shrink_target))
    p_decide = lam * p_model + (1.0 - lam) * 0.5

    p_mkt_a = implied_prob_from_american(odds_a)
    gap_ok = abs(p_decide - p_mkt_a) >= (prob_gap_pp / 100.0)

    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    ev_a_dec = ((p_decide * dec_a) - 1.0) * 100.0
    ev_b_dec = (((1.0 - p_decide) * dec_b) - 1.0) * 100.0

    reasons = []
    if not gap_ok:
        reasons.append(f"prob gap < {prob_gap_pp}pp")

    best_side = None
    best_ev = max(ev_a_dec, ev_b_dec)
    if best_ev < min_edge_pct:
        reasons.append(f"edge < {min_edge_pct}%")
    else:
        best_side = "A" if ev_a_dec >= ev_b_dec else "B"

    reason = " & ".join(reasons) if reasons else "passes filters"
    return {
        "p_decide": p_decide,
        "ev_a_dec": ev_a_dec,
        "ev_b_dec": ev_b_dec,
        "choice": best_side,
        "reason": reason,
    }


def decide_bo2_3way(
    p_map_model: float,
    n_matches_a: int,
    n_matches_b: int,
    min_edge_pct: float,
    prob_gap_pp: float,
    shrink_target: int,
    draw_k: float,
    odds_a20: float,
    odds_draw: float,
    odds_b02: float,
):
    """3-way decision for BO2. Returns dict with selected outcome/EV/prob/odds + reason."""
    eff_matches = min(n_matches_a, n_matches_b)
    lam = min(1.0, eff_matches / float(shrink_target))
    p_map_decide = lam * p_map_model + (1.0 - lam) * 0.5

    probs = compute_bo2_probs(p_map_decide, k=draw_k)
    evs = {
        "A2-0": ev_pct_decimal(probs["A2-0"], odds_a20),
        "DRAW": ev_pct_decimal(probs["DRAW"], odds_draw),
        "B0-2": ev_pct_decimal(probs["B0-2"], odds_b02),
    }
    selected = max(evs.items(), key=lambda kv: kv[1])
    sel_outcome, sel_ev = selected[0], selected[1]
    sel_prob = probs[sel_outcome]
    sel_odds = {"A2-0": odds_a20, "DRAW": odds_draw, "B0-2": odds_b02}[sel_outcome]

    imp = {}
    vals = [x for x in [odds_a20, odds_draw, odds_b02] if x and x > 1.0]
    if vals:
        inv = [1.0 / x for x in vals]
        s = sum(inv)
        inv_map = {
            "A2-0": (1.0 / odds_a20) if odds_a20 > 1.0 else 0.0,
            "DRAW": (1.0 / odds_draw) if odds_draw > 1.0 else 0.0,
            "B0-2": (1.0 / odds_b02) if odds_b02 > 1.0 else 0.0,
        }
        imp = {k: (v / s if s > 0 else 0.0) for k, v in inv_map.items()}
    p_mkt_sel = imp.get(sel_outcome, 0.0)

    reasons = []
    if sel_ev < min_edge_pct:
        reasons.append(f"edge < {min_edge_pct}%")
    if abs(sel_prob - p_mkt_sel) < (prob_gap_pp / 100.0):
        reasons.append(f"prob gap < {prob_gap_pp}pp")

    ok = len(reasons) == 0
    reason = " & ".join(reasons) if reasons else "passes filters"

    return {
        "p_map_decide": p_map_decide,
        "probs": probs,
        "evs": evs,
        "selected_outcome": sel_outcome if ok else None,
        "selected_prob": sel_prob if ok else None,
        "selected_odds": sel_odds if ok else None,
        "selected_ev_pct": sel_ev if ok else None,
        "reason": reason,
    }
