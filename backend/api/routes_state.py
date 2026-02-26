"""
State API: current State and Derived (placeholder stub).
"""
from fastapi import APIRouter

from engine.models import Config, Derived, Frame, State

router = APIRouter(prefix="/state", tags=["state"])


@router.get("/current")
def get_state_current() -> dict:
    """
    Return current authoritative state and derived outputs.
    Stub: returns static placeholder matching State/Derived shape.
    """
    state = State(
        config=Config(),
        last_frame=Frame(),
        team_mapping={},
        map_index=0,
        last_total_rounds=0,
    )
    derived = Derived(p_hat=0.5, rail_low=0.0, rail_high=1.0, bound_low=0.0, bound_high=1.0, kappa=0.0, debug={})
    return {
        "state": {
            "config": {
                "source": state.config.source,
                "match_id": state.config.match_id,
                "poll_interval_s": state.config.poll_interval_s,
                "lock_team_mapping": state.config.lock_team_mapping,
                "market_delay_s": state.config.market_delay_s,
            },
            "map_index": state.map_index,
            "last_total_rounds": state.last_total_rounds,
            "team_mapping": state.team_mapping,
            "last_frame": (
                {
                    "timestamp": state.last_frame.timestamp,
                    "teams": list(state.last_frame.teams),
                    "scores": list(state.last_frame.scores),
                }
                if state.last_frame
                else None
            ),
        },
        "derived": {
            "p_hat": derived.p_hat,
            "rail_low": derived.rail_low,
            "rail_high": derived.rail_high,
            "bound_low": derived.bound_low,
            "bound_high": derived.bound_high,
            "kappa": derived.kappa,
            "debug": derived.debug,
        },
    }
