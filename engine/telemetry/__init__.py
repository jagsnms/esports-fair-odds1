"""
Source-agnostic telemetry scaffolding: core types, monotonic gate, match context, selector.
No FastAPI or backend dependencies.
"""
from engine.telemetry.core import (
    CanonicalFrameEnvelope,
    IdentityEntry,
    MatchContext,
    SourceHealth,
    SourceKind,
)
from engine.telemetry.monotonic import (
    MonotonicKey,
    compute_monotonic_key_from_bo3_snapshot,
    compute_monotonic_key_from_grid_state,
    should_accept,
)
from engine.telemetry.selector import (
    SourceDecision,
    SourceSelectorPolicy,
    decide,
    default_source_selector_policy,
    is_fresh,
    maybe_switch,
)
from engine.telemetry.session import SessionKey, SessionRegistry, SessionRuntime

__all__ = [
    "CanonicalFrameEnvelope",
    "IdentityEntry",
    "MatchContext",
    "MonotonicKey",
    "SourceDecision",
    "SourceHealth",
    "SourceKind",
    "SourceSelectorPolicy",
    "compute_monotonic_key_from_bo3_snapshot",
    "compute_monotonic_key_from_grid_state",
    "decide",
    "default_source_selector_policy",
    "is_fresh",
    "SessionKey",
    "SessionRegistry",
    "SessionRuntime",
    "maybe_switch",
    "should_accept",
]
