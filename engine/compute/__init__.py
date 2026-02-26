"""Compute: bounds, rails, resolve_p_hat. Pure functions from frame/state/config."""

from engine.compute.bounds import compute_bounds
from engine.compute.rails import compute_rails
from engine.compute.resolve import resolve_p_hat

__all__ = ["compute_bounds", "compute_rails", "resolve_p_hat"]
