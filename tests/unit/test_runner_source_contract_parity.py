"""
Stage 3: Assert that when config.invariant_diagnostics is True, resolve receives it and emits contract_diagnostics.
Runner wiring: setattr(config, "contract_testing_mode", getattr(config, "invariant_diagnostics", False)) before resolve.
"""
from __future__ import annotations

from engine.models import Config
from engine.compute.resolve import resolve_p_hat
from engine.models import Frame, State


def test_invariant_diagnostics_wired_to_contract_diagnostics() -> None:
    """Config.invariant_diagnostics=True causes resolve_p_hat to return contract_diagnostics in debug."""
    config = Config(invariant_diagnostics=True)
    setattr(config, "contract_testing_mode", getattr(config, "invariant_diagnostics", False))
    frame = Frame(
        timestamp=1000.0,
        teams=("A", "B"),
        scores=(1, 0),
        alive_counts=(5, 4),
        hp_totals=(400.0, 300.0),
        loadout_totals=(8000.0, 6000.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
    )
    state = State(config=config)
    rails = (0.2, 0.8)
    p_hat, dbg = resolve_p_hat(frame, config, state, rails)
    assert "contract_diagnostics" in dbg
    assert dbg["contract_diagnostics"].get("contract_testing_mode") is True
