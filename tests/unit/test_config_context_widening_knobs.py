from __future__ import annotations

from engine.config import merge_config
from engine.models import Config


def test_merge_config_context_widen_knobs_coerce_and_clamp() -> None:
    current = Config()
    merged = merge_config(
        current,
        {
            "context_widening_enabled": 1,
            "context_widen_beta": 9.0,
            "uncertainty_mult_min": 1.2,
            "uncertainty_mult_max": 1.1,
            "context_risk_weight_leverage": 2.0,
            "context_risk_weight_fragility": 1.0,
            "context_risk_weight_missingness": 1.0,
        },
    )
    assert merged.context_widening_enabled is True
    assert merged.context_widen_beta == 2.0
    assert merged.uncertainty_mult_min == 1.2
    assert merged.uncertainty_mult_max == 1.2
    assert abs(merged.context_risk_weight_leverage - 0.5) <= 1e-9
    assert abs(merged.context_risk_weight_fragility - 0.25) <= 1e-9
    assert abs(merged.context_risk_weight_missingness - 0.25) <= 1e-9


def test_merge_config_context_weights_fallback_when_zeroed() -> None:
    current = Config()
    merged = merge_config(
        current,
        {
            "context_risk_weight_leverage": 0.0,
            "context_risk_weight_fragility": 0.0,
            "context_risk_weight_missingness": 0.0,
        },
    )
    assert abs(merged.context_risk_weight_leverage - 0.4) <= 1e-9
    assert abs(merged.context_risk_weight_fragility - 0.4) <= 1e-9
    assert abs(merged.context_risk_weight_missingness - 0.2) <= 1e-9
