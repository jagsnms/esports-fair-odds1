"""
CS2 microstate adjustment for p_hat: alive, HP, econ, bomb (bounded).
Returns adjustment in [-0.08, +0.08]. Defensive when fields missing.
No midround mixture; no new dependencies.
"""
from __future__ import annotations

from engine.models import Frame

# Component caps (before sum); final sum clamped to [-ADJ_CAP, +ADJ_CAP]
ADJ_CAP = 0.08
ALIVE_PER_PLAYER = 0.02
ALIVE_CAP = 0.06  # ~3 alive diff
HP_SCALE = 500.0   # hp diff / HP_SCALE * HP_CAP -> max HP_CAP
HP_CAP = 0.03
ECON_SCALE = 20000.0
ECON_CAP = 0.03
# Bomb: up to 0.02 only if we can infer which team benefits; Frame has no attacker_side, so skip
BOMB_CAP = 0.02

BUY_TIME_PHASES = frozenset({"BUY_TIME", "FREEZETIME"})


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round_phase_upper(frame: Frame) -> str | None:
    """Read round_phase from frame.bomb_phase_time_remaining (round_phase or phase), normalize uppercase."""
    bomb = getattr(frame, "bomb_phase_time_remaining", None)
    if not isinstance(bomb, dict):
        return None
    rp = bomb.get("round_phase") or bomb.get("phase")
    if rp is None:
        return None
    return str(rp).strip().upper()


def micro_adjustment_cs2(frame: Frame) -> float:
    """
    Compute a small adjustment in [-0.08, +0.08] from alive/HP/econ/bomb.
    Positive = team A favored. Defensive: missing/zeroed fields -> 0 for that component.
    In BUY_TIME/FREEZETIME: skip econ (loadout) adjustment to avoid fake loadout gaps.
    """
    total = 0.0
    phase_upper = _round_phase_upper(frame)
    skip_econ = phase_upper in BUY_TIME_PHASES

    # Alive diff: ~0.02 per alive
    alive = getattr(frame, "alive_counts", None)
    if isinstance(alive, (tuple, list)) and len(alive) >= 2:
        a = int(alive[0]) if alive[0] is not None else 0
        b = int(alive[1]) if alive[1] is not None else 0
        diff = a - b
        total += _clip(diff * ALIVE_PER_PLAYER, -ALIVE_CAP, ALIVE_CAP)

    # HP diff: up to ~0.03
    hp = getattr(frame, "hp_totals", None)
    if isinstance(hp, (tuple, list)) and len(hp) >= 2:
        ha = float(hp[0]) if hp[0] is not None else 0.0
        hb = float(hp[1]) if hp[1] is not None else 0.0
        if HP_SCALE > 0:
            raw = (ha - hb) / HP_SCALE * HP_CAP
            total += _clip(raw, -HP_CAP, HP_CAP)

    # Econ (alive loadout equipment value) diff: up to ~0.03 (skipped in BUY_TIME/FREEZETIME)
    if not skip_econ:
        econ = getattr(frame, "loadout_totals", None)
        if isinstance(econ, (tuple, list)) and len(econ) >= 2:
            ea = float(econ[0]) if econ[0] is not None else 0.0
            eb = float(econ[1]) if econ[1] is not None else 0.0
            if ECON_SCALE > 0:
                raw = (ea - eb) / ECON_SCALE * ECON_CAP
                total += _clip(raw, -ECON_CAP, ECON_CAP)

    # Bomb: only if we can infer which team benefits; Frame typically has no attacker_side -> skip
    # If bomb_phase has attacker_side / planter_side we could add ±BOMB_CAP; for now 0.

    return _clip(total, -ADJ_CAP, ADJ_CAP)


def micro_adjustment_cs2_breakdown(frame: Frame) -> dict[str, float]:
    """
    Return component breakdown for explain logging: alive_adj, hp_adj, econ_adj.
    Each component is in [-cap, +cap]; sum is clipped in micro_adjustment_cs2.
    """
    alive_adj = 0.0
    hp_adj = 0.0
    econ_adj = 0.0
    phase_upper = _round_phase_upper(frame)
    skip_econ = phase_upper in BUY_TIME_PHASES

    alive = getattr(frame, "alive_counts", None)
    if isinstance(alive, (tuple, list)) and len(alive) >= 2:
        a = int(alive[0]) if alive[0] is not None else 0
        b = int(alive[1]) if alive[1] is not None else 0
        diff = a - b
        alive_adj = _clip(diff * ALIVE_PER_PLAYER, -ALIVE_CAP, ALIVE_CAP)

    hp = getattr(frame, "hp_totals", None)
    if isinstance(hp, (tuple, list)) and len(hp) >= 2:
        ha = float(hp[0]) if hp[0] is not None else 0.0
        hb = float(hp[1]) if hp[1] is not None else 0.0
        if HP_SCALE > 0:
            raw = (ha - hb) / HP_SCALE * HP_CAP
            hp_adj = _clip(raw, -HP_CAP, HP_CAP)

    if not skip_econ:
        econ = getattr(frame, "loadout_totals", None)
        if isinstance(econ, (tuple, list)) and len(econ) >= 2:
            ea = float(econ[0]) if econ[0] is not None else 0.0
            eb = float(econ[1]) if econ[1] is not None else 0.0
            if ECON_SCALE > 0:
                raw = (ea - eb) / ECON_SCALE * ECON_CAP
                econ_adj = _clip(raw, -ECON_CAP, ECON_CAP)

    return {"alive_adj": alive_adj, "hp_adj": hp_adj, "econ_adj": econ_adj}
