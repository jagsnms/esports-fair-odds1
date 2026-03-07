"""Compute: bounds, rails, resolve_p_hat, movement confidence. Pure functions from frame/state/config."""

from engine.compute.bounds import compute_bounds
from engine.compute.rails import compute_rails
from engine.compute.resolve import compute_movement_confidence, resolve_p_hat

__all__ = ["compute_bounds", "compute_rails", "compute_movement_confidence", "resolve_p_hat"]
