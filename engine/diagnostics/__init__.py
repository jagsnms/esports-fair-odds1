"""Diagnostics: invariant checks and debug helpers."""

from engine.diagnostics.invariants import (
    check_monotonic_time,
    check_prob_range,
    check_team_identity_consistency,
)

__all__ = ["check_prob_range", "check_monotonic_time", "check_team_identity_consistency"]
