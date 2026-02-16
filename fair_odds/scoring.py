"""Match scoring with tier-based and recency adjustments."""

import pandas as pd

from .data import get_team_tier, piecewise_recent_weights

USE_SERIES_SCORE_MOD = True
SERIES_CLAMP = 1.25
SERIES_WEIGHTS = {0: 0.25, 1: 0.50, 2: 0.75, 3: 1.00, 4: 1.25}
SERIES_PCT_OF_BASE_CAP = 0.40


def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def base_points_from_tier(win: bool, tier: float) -> float:
    if win:
        if tier == 1:
            return 4.0
        elif tier == 1.5:
            return 3.0
        elif tier == 2:
            return 2.5
        elif tier == 3:
            return 2.0
        elif tier == 4:
            return 1.5
        else:
            return 1.0
    else:
        if tier == 1:
            return -1.0
        elif tier == 1.5:
            return -1.5
        elif tier == 2:
            return -2.0
        elif tier == 3:
            return -2.5
        elif tier == 4:
            return -3.0
        else:
            return -4.0


def normalize_series_result_from_fields(match: dict) -> str:
    s = str(match.get("series", "") or match.get("series_result", "")).strip()
    if s in {"2-0", "2-1", "1-2", "0-2"}:
        return s
    key_pairs = [
        ("us", "them"),
        ("series_us", "series_them"),
        ("maps_us", "maps_them"),
        ("score_us", "score_them"),
    ]
    us = them = None
    for a, b in key_pairs:
        if a in match or b in match:
            us = _to_int(match.get(a))
            them = _to_int(match.get(b))
            break
    if us is None or them is None:
        return ""
    if us == them:
        return ""
    if (us, them) == (2, 0):
        return "2-0"
    if (us, them) == (2, 1):
        return "2-1"
    if (us, them) == (1, 2):
        return "1-2"
    if (us, them) == (0, 2):
        return "0-2"
    return "2-1" if us > them else "1-2"


def series_score_modifier_5tier(
    team_tier: float,
    opp_tier: float,
    result: str,
    clamp: float = SERIES_CLAMP,
    weights: dict = SERIES_WEIGHTS,
) -> float:
    if result not in {"2-0", "2-1", "1-2", "0-2"}:
        return 0.0
    gap = float(opp_tier) - float(team_tier)
    abs_gap = int(min(4, abs(gap)))
    w = float(weights.get(abs_gap, list(weights.values())[-1]))
    if gap >= 2:
        table = {"2-0": 0.0, "2-1": -w, "1-2": -w, "0-2": -w}
        return max(-clamp, min(clamp, table[result]))
    if gap == 1:
        table = {"2-0": 0.25, "2-1": -0.50, "1-2": -0.50, "0-2": -0.75}
        return max(-clamp, min(clamp, table[result]))
    if gap == 0:
        table = {"2-0": 0.25, "2-1": 0.0, "1-2": -0.0, "0-2": -0.25}
        return max(-clamp, min(clamp, table[result]))
    if result == "0-2":
        return 0.0
    if result == "1-2":
        val = 0.50 if abs_gap == 1 else w
        return max(-clamp, min(clamp, val))
    if result in ("2-1", "2-0"):
        val = min(w + 0.25, clamp)
        return max(-clamp, min(clamp, val))
    return 0.0


def calculate_score(
    matches,
    df: pd.DataFrame,
    current_opponent_tier=None,
    weight_scheme: str = "piecewise",
    K: int = 6,
    decay: float = 0.85,
    floor: float = 0.6,
    newest_first: bool = True,
    draw_policy: str = "graded",
    self_team_tier: float | None = None,
    draw_gamma: float = 0.5,
    draw_gap_cap: float = 3.0,
    draw_gap_power: float = 1.0,
    use_series_mod: bool = True,
    series_clamp: float = SERIES_CLAMP,
    series_weights: dict = SERIES_WEIGHTS,
    series_pct_cap: float = SERIES_PCT_OF_BASE_CAP,
):
    raw_score = 0.0
    adjusted_score = 0.0
    breakdown = []
    n = len(matches)
    if weight_scheme == "piecewise":
        weights = piecewise_recent_weights(
            n, K=K, decay=decay, floor=floor, newest_first=newest_first
        )
    else:
        weights = [1.0] * n

    for i, match in enumerate(matches):
        opp = match["opponent"]
        win = bool(match.get("win", False))
        draw_flag = bool(match.get("draw", False))
        tier = get_team_tier(opp, df)

        if draw_flag:
            if draw_policy == "neutral":
                points = 0.0
                base_txt = "Draw (neutral 0)"
            elif draw_policy == "loss":
                points = base_points_from_tier(False, tier)
                base_txt = f"Draw→Loss rule ({points:+.2f})"
            else:
                my_tier = float(self_team_tier) if self_team_tier is not None else 3.0
                rel = my_tier - tier
                sign = 1.0 if rel > 0 else (-1.0 if rel < 0 else 0.0)
                win_mag = base_points_from_tier(True, tier)
                loss_mag = abs(base_points_from_tier(False, tier))
                base_mag = 0.5 * (win_mag + loss_mag)
                gap = min(abs(rel), draw_gap_cap) / max(draw_gap_cap, 1e-9)
                gap = gap ** max(draw_gap_power, 1e-9)
                points = sign * draw_gamma * base_mag * gap
                base_txt = f"Draw (graded {points:+.2f}; rel_gap={rel:.1f})"
        else:
            points = (
                base_points_from_tier(True, tier) if win else base_points_from_tier(False, tier)
            )
            base_txt = "Win" if win else "Loss"
            if use_series_mod:
                series_res = normalize_series_result_from_fields(match)
                if series_res in {"2-0", "2-1", "1-2", "0-2"}:
                    my_tier = float(self_team_tier) if self_team_tier is not None else 3.0
                    s_bump = series_score_modifier_5tier(
                        team_tier=my_tier,
                        opp_tier=tier,
                        result=series_res,
                        clamp=series_clamp,
                        weights=series_weights,
                    )
                    if s_bump:
                        pct_cap = series_pct_cap * max(0.5, abs(points))
                        s_bump = max(-pct_cap, min(pct_cap, s_bump))
                        points += s_bump
                        base_txt += f" + SeriesMod({series_res} {s_bump:+.2f})"

        raw_score += points

        if current_opponent_tier is not None:
            tier_gap = tier - current_opponent_tier
            positiveish = win or (draw_flag and draw_policy != "loss" and points >= 0)
            if positiveish:
                weight_tier_old = (
                    1 + min(0.4, abs(tier_gap) * 0.2)
                    if tier_gap < 0
                    else 1 - min(0.3, tier_gap * 0.15)
                )
            else:
                weight_tier_old = (
                    1 - min(0.2, abs(tier_gap) * 0.1)
                    if tier_gap < 0
                    else 1 + min(0.5, tier_gap * 0.25)
                )
            weight_tier_old = max(0.5, min(weight_tier_old, 1.5))
        else:
            tier_gap = 0.0
            weight_tier_old = 1.0

        weight_tier = 1 + (weight_tier_old - 1) * 0.3
        weight_tier = max(0.85, min(weight_tier, 1.15))
        w_match = weights[i]
        adj_points = points * weight_tier * w_match
        adjusted_score += adj_points
        breakdown.append(
            f"{base_txt} vs {opp} (OppTier {tier}, CurOppGap {tier_gap:.1f}) "
            f"Pts={points:+.2f} × TierW={weight_tier:.2f} × RecW={w_match:.3f} = {adj_points:+.3f}"
        )
    return raw_score, adjusted_score, breakdown
