from __future__ import annotations

import inspect

import numpy as np

from ppg_hr.experimental.cascade_solver import aggregate_metric_arrays
from ppg_hr.experimental.fusion import fuse_final_hr
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams


def test_fusion_signature_does_not_accept_reference_hr() -> None:
    params = inspect.signature(fuse_final_hr).parameters
    assert "ref_hr_bpm" not in params
    assert "reference_hr_bpm" not in params


def test_recovery_grace_does_not_immediately_force_baseline() -> None:
    result = fuse_final_hr(
        time_s=np.asarray([0.0, 1.0, 2.0, 3.0, 5.0]),
        baseline_hr_bpm=np.asarray([70.0, 70.0, 70.0, 75.0, 80.0]),
        adaptive_hr_bpm=np.asarray([90.0, 91.0, 92.0, 80.0, 120.0]),
        segment_label=np.asarray(["rest", "motion", "motion", "recovery", "recovery"], dtype=object),
        qc_status=np.asarray(["ok", "ok", "ok", "ok", "ok"], dtype=object),
        adaptive_filter="lms",
        motion_end_s=2.0,
        params=ProtocolTrialParams(Recovery_Grace_S=2.0, Recovery_Diff_Bpm=10.0),
    )

    assert result.final_hr_bpm.tolist() == [70.0, 91.0, 92.0, 80.0, 80.0]
    assert result.final_source.tolist() == [
        "baseline_fft",
        "adaptive_lms",
        "adaptive_lms",
        "adaptive_lms",
        "recovery_fallback",
    ]
    assert "recovery_grace" in result.fusion_reason[3]
    assert "recovery_diff" in result.fusion_reason[4]


def test_qc_fallback_uses_baseline_without_reference_hr() -> None:
    result = fuse_final_hr(
        time_s=np.asarray([1.0]),
        baseline_hr_bpm=np.asarray([72.0]),
        adaptive_hr_bpm=np.asarray([110.0]),
        segment_label=np.asarray(["motion"], dtype=object),
        qc_status=np.asarray(["fallback_baseline"], dtype=object),
        adaptive_filter="rff_lms",
        motion_end_s=np.nan,
        params=ProtocolTrialParams(),
    )

    assert result.final_hr_bpm.tolist() == [72.0]
    assert result.final_source.tolist() == ["qc_fallback_baseline"]
    assert "qc_status=fallback_baseline" in result.fusion_reason[0]


def test_deployment_global_uses_baseline_rest_adaptive_motion_and_fused_recovery() -> None:
    result = fuse_final_hr(
        time_s=np.asarray([0.0, 1.0, 2.0, 12.0]),
        baseline_hr_bpm=np.asarray([70.0, 70.0, 70.0, 75.0]),
        adaptive_hr_bpm=np.asarray([95.0, 100.0, 101.0, 120.0]),
        segment_label=np.asarray(["rest", "motion", "motion", "recovery"], dtype=object),
        qc_status=np.asarray(["ok", "ok", "ok", "ok"], dtype=object),
        adaptive_filter="lms",
        motion_end_s=2.0,
        target_scope="global",
        params=ProtocolTrialParams(
            global_objective_strategy="deployment_global",
            Recovery_Grace_S=2.0,
            Recovery_Diff_Bpm=10.0,
        ),
    )

    assert result.final_hr_bpm.tolist() == [70.0, 100.0, 101.0, 75.0]
    assert result.final_source.tolist() == [
        "baseline_fft",
        "adaptive_lms",
        "adaptive_lms",
        "recovery_fallback",
    ]
    assert "deployment_global" in result.fusion_reason[0]


def test_aggregate_metric_arrays_reports_final_metrics() -> None:
    arrays = {
        "ref_hr_bpm": np.asarray([70.0, 100.0]),
        "baseline_hr_bpm": np.asarray([70.0, 70.0]),
        "adaptive_hr_bpm": np.asarray([100.0, 100.0]),
        "final_hr_bpm": np.asarray([70.0, 100.0]),
        "filtered_mask": np.asarray([True, True]),
    }

    metrics = aggregate_metric_arrays(arrays)

    assert metrics["baseline_aae_bpm"] == 15.0
    assert metrics["adaptive_aae_bpm"] == 15.0
    assert metrics["final_aae_bpm"] == 0.0
    assert metrics["final_acc_pct"] == 100.0
