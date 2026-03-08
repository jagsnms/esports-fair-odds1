from __future__ import annotations

from tools.calibration_reliability_evidence_gate import (
    SCHEMA_VERSION,
    build_calibration_reliability_summary,
)


def _bins(a: float, b: float) -> list[dict]:
    return [
        {"bin_index": 0, "count": 5, "mean_prediction": a, "empirical_rate": b},
        {"bin_index": 1, "count": 7, "mean_prediction": a + 0.1, "empirical_rate": b + 0.1},
    ]


def test_valid_pass_case_has_shared_schema_and_deterministic_pairs() -> None:
    records = [
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "current",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.21,
            "log_loss": 0.62,
            "reliability_curve_bins": _bins(0.52, 0.50),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": 0.24,
            "log_loss": 0.66,
            "reliability_curve_bins": _bins(0.51, 0.48),
        },
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "baseline",
            "seed": 99,
            "segment": "late",
            "brier_score": 0.23,
            "log_loss": 0.64,
            "reliability_curve_bins": _bins(0.50, 0.49),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "current",
            "seed": None,
            "segment": None,
            "brier_score": 0.22,
            "log_loss": 0.63,
            "reliability_curve_bins": _bins(0.54, 0.52),
        },
    ]
    summary_a = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=records,
        generated_at="2026-03-08T00:00:00Z",
    )
    summary_b = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=list(reversed(records)),
        generated_at="2026-03-08T00:00:00Z",
    )
    assert summary_a == summary_b
    assert summary_a["schema_version"] == SCHEMA_VERSION
    assert summary_a["gate_status"] == "pass"
    assert summary_a["incomplete_reasons"] == []
    assert len(summary_a["comparison_pairs"]) == 2
    for pair in summary_a["comparison_pairs"]:
        assert "baseline_metrics" in pair
        assert "current_metrics" in pair
        assert set(pair["baseline_metrics"]) == {"brier_score", "log_loss", "reliability_curve_bins"}
        assert set(pair["current_metrics"]) == {"brier_score", "log_loss", "reliability_curve_bins"}
        delta = pair["delta"]
        assert set(delta) == {"brier_score_delta", "log_loss_delta", "reliability_curve_bins_delta"}


def test_incomplete_evidence_when_missing_required_slice_and_metric() -> None:
    records = [
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": 0.24,
            "log_loss": 0.66,
            "reliability_curve_bins": _bins(0.51, 0.48),
        },
        {
            "evidence_source": "replay",
            "dataset_id": "replay_set_a",
            "evaluation_scope": "current",
            "seed": None,
            "segment": None,
            "brier_score": 0.22,
            # log_loss intentionally missing to trigger incomplete_evidence
            "reliability_curve_bins": _bins(0.54, 0.52),
        },
        {
            "evidence_source": "simulation",
            "dataset_id": "sim_seed_99",
            "evaluation_scope": "baseline",
            "seed": 99,
            "segment": None,
            "brier_score": 0.23,
            "log_loss": 0.64,
            "reliability_curve_bins": _bins(0.50, 0.49),
        },
    ]
    summary = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=records,
        generated_at="2026-03-08T00:00:00Z",
    )
    assert summary["gate_status"] == "incomplete_evidence"
    reasons = " | ".join(summary["incomplete_reasons"])
    assert "missing metric key log_loss" in reasons
    assert "required slice missing: source=simulation, scope=current" in reasons


def test_fail_on_malformed_enum_and_metric_types() -> None:
    records = [
        {
            "evidence_source": "replayish",  # invalid enum
            "dataset_id": "bad_record",
            "evaluation_scope": "baseline",
            "seed": None,
            "segment": None,
            "brier_score": "0.24",  # invalid type
            "log_loss": 0.66,
            "reliability_curve_bins": "not-an-array",  # invalid type
        }
    ]
    summary = build_calibration_reliability_summary(
        baseline_ref="baseline:report-001",
        current_ref="current:report-002",
        evidence_records=records,
        generated_at="2026-03-08T00:00:00Z",
    )
    assert summary["gate_status"] == "fail"
    reasons = " | ".join(summary["incomplete_reasons"])
    assert "evidence_source invalid enum" in reasons
    assert ".brier_score invalid type/value" in reasons
    assert ".reliability_curve_bins invalid type/value" in reasons
