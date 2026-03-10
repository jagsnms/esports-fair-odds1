"""
Reduce prev state + frame + config -> new State. Identity lock and transition segmentation.
"""
from __future__ import annotations

from engine.models import Config, Frame, State


def _team_one_key(frame: Frame, config: Config) -> str:
    """Prefer provider_id else name (team_one = team A when team_a_is_team_one else team B)."""
    pid = getattr(frame, "team_one_provider_id", None)
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    teams = getattr(frame, "teams", ("", ""))
    if getattr(config, "team_a_is_team_one", True):
        return teams[0] if len(teams) > 0 else "team_one"
    return teams[1] if len(teams) > 1 else "team_two"


def _team_two_key(frame: Frame, config: Config) -> str:
    """Prefer provider_id else name (team_two = team B when team_a_is_team_one else team A)."""
    pid = getattr(frame, "team_two_provider_id", None)
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    teams = getattr(frame, "teams", ("", ""))
    if getattr(config, "team_a_is_team_one", True):
        return teams[1] if len(teams) > 1 else "team_two"
    return teams[0] if len(teams) > 0 else "team_one"


def reduce_state(prev: State, frame: Frame, config: Config) -> State:
    """
    Update last_frame, map_index, last_total_rounds; apply identity lock and transition detection.
    - team_mapping: if not locked, set from config + frame keys; if locked and empty, init once; else keep.
    - segment_id: increment when map_index or series_score changes; then update last_map_index, last_series_score.
    """
    total_rounds = frame.scores[0] + frame.scores[1]
    new_map_index = frame.map_index
    new_series_score = tuple(frame.series_score)

    # Team mapping: build keys from frame (provider_id or name)
    team_one_key = _team_one_key(frame, config)
    team_two_key = _team_two_key(frame, config)
    a_is_team_one = getattr(config, "team_a_is_team_one", True)

    if not getattr(config, "lock_team_mapping", False):
        team_mapping = {
            "a_is_team_one": a_is_team_one,
            "team_one_key": team_one_key,
            "team_two_key": team_two_key,
        }
    else:
        existing = prev.team_mapping
        if not existing or not isinstance(existing, dict):
            team_mapping = {
                "a_is_team_one": a_is_team_one,
                "team_one_key": team_one_key,
                "team_two_key": team_two_key,
            }
        else:
            team_mapping = dict(existing)

    # Transition detection: bump segment_id when map or series changes
    segment_id = prev.segment_id
    if prev.last_map_index is not None and new_map_index != prev.last_map_index:
        segment_id += 1
    if prev.last_series_score is not None and new_series_score != prev.last_series_score:
        segment_id += 1

    return State(
        config=config,
        last_frame=frame,
        team_mapping=team_mapping,
        map_index=new_map_index,
        last_total_rounds=total_rounds,
        segment_id=segment_id,
        last_series_score=new_series_score,
        last_map_index=new_map_index,
    )
