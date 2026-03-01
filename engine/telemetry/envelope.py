"""
Canonical envelope processing: monotonic accept/reject, health updates, active_source bootstrap.
Pure Python; no FastAPI or backend dependencies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.telemetry.core import CanonicalFrameEnvelope, MatchContext, SourceKind


def process_canonical_envelope(ctx: "MatchContext", env: "CanonicalFrameEnvelope") -> tuple[bool, str | None]:
    """
    Source-agnostic envelope gate: validate key, run should_accept, update ctx and SourceHealth.
    Does NOT call reduce/compute. Returns (accepted: bool, reject_reason: str | None).
    """
    from engine.telemetry.core import SourceKind
    from engine.telemetry.monotonic import should_accept

    source = env.source
    now = env.observed_ts
    health = ctx.get_or_create_source_health(source)

    if env.key is None:
        ctx.rejected_count += 1
        ctx.last_reject_reason = "missing_key"
        health.err_count += 1
        health.last_err_ts = now
        health.last_reason = "missing_key"
        return (False, "missing_key")

    accept, reject_reason = should_accept(ctx.last_accepted_key, env.key, reject_missing_key=True)
    if not accept:
        ctx.rejected_count += 1
        ctx.last_reject_reason = reject_reason
        health.err_count += 1
        health.last_err_ts = now
        health.last_reason = reject_reason
        return (False, reject_reason)

    ctx.last_accepted_key = env.key
    ctx.accepted_count += 1
    health.ok_count += 1
    health.last_ok_ts = now
    health.last_reason = None

    # Bootstrap active_source when first frame accepted (e.g. BO3)
    if getattr(ctx, "active_source", None) is None:
        ctx.active_source = source

    # Last accepted envelope summary for diagnostics
    ctx.last_accepted_env_summary = {
        "source": getattr(source, "name", str(source)),
        "match_id": env.match_id,
        "key_display": env.key.to_display(),
        "observed_ts": env.observed_ts,
    }
    return (True, None)
