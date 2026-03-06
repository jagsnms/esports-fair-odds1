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
    contract_diag = dbg["contract_diagnostics"]
    assert contract_diag.get("contract_testing_mode") is True
    assert contract_diag.get("rail_low") == 0.2
    assert contract_diag.get("rail_high") == 0.8
    assert contract_diag.get("p_hat_prev") is not None
    assert contract_diag.get("p_hat_final") == p_hat
    assert contract_diag.get("round_time_remaining_s") is None
    assert contract_diag.get("alive_counts") == (5, 4)
    assert contract_diag.get("hp_totals") == (400.0, 300.0)
    assert contract_diag.get("loadout_totals") == (8000.0, 6000.0)
    assert contract_diag.get("round_phase") == "IN_PROGRESS"
    assert contract_diag.get("round_number") is None
