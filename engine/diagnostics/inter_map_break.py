"""
Heuristics to detect inter-map break (map ended, next map not started yet).
Used by runner to keep corridors/p_hat stable instead of collapsing to zeros.
"""
from __future__ import annotations

from typing import Tuple

from engine.compute.rails_cs2 import _cs2_win_target
from engine.models import Frame, State


def detect_inter_map_break(frame: Frame, state: State) -> Tuple[bool, str]:
  """
  Return (is_break, reason) when we believe we are between maps.

  Heuristics:
  1) round_phase in a known break/finished set.
  2) Map appears decided by score AND microstate missing.
  3) Scores reset to (0,0) while series_score / map_index do not indicate a new map.
  """
  # 1) round_phase from bomb_phase_time_remaining, when available.
  bomb = getattr(frame, "bomb_phase_time_remaining", None)
  if isinstance(bomb, dict):
      phase = bomb.get("round_phase") or bomb.get("phase")
      if isinstance(phase, str):
          upper = phase.upper()
          if upper in {"POSTGAME", "FINISHED", "MAP_END", "INTERMISSION"}:
              return True, f"round_phase={upper}"

  scores = getattr(frame, "scores", (0, 0))
  ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
  rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
  wt = _cs2_win_target(ra, rb)

  # Microstate presence: treat as missing if alive is 0/0 and no loadout_totals and hp_totals ~ 0/0.
  alive = getattr(frame, "alive_counts", (0, 0))
  loadout_totals = getattr(frame, "loadout_totals", None)
  hp_totals = getattr(frame, "hp_totals", (0.0, 0.0))
  alive_missing = (alive is None) or (len(alive) >= 2 and (alive[0] == 0 and alive[1] == 0))
  loadout_missing = loadout_totals is None
  hp_missing = len(hp_totals) >= 2 and (hp_totals[0] == 0.0 and hp_totals[1] == 0.0)
  micro_missing = alive_missing and loadout_missing and hp_missing

  # 2) Map decided by score and microstate missing.
  if (ra >= wt or rb >= wt) and micro_missing:
      return True, f"map_decided_scores_no_microstate (scores={ra}-{rb}, wt={wt})"

  # 3) Scores reset to (0,0) while series/map do not indicate a new map started.
  series_score = getattr(frame, "series_score", (0, 0))
  last_series_score = getattr(state, "last_series_score", None)
  map_index = getattr(frame, "map_index", 0)
  last_map_index = getattr(state, "last_map_index", None)
  last_total_rounds = getattr(state, "last_total_rounds", 0)

  if ra == 0 and rb == 0 and last_total_rounds > 0:
      if last_series_score is not None and tuple(last_series_score) == tuple(series_score):
          if last_map_index is not None and last_map_index == map_index:
              return True, f"scores_reset_without_series_or_map_advance (last_total_rounds={last_total_rounds})"

  return False, ""

