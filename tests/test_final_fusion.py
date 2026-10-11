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
        segment_label=np.asarray(
            ["rest", "motion", "motion", "recovery", "recovery"], dtype=object
        ),
        qc_status=np.asarray(["ok", "ok", "ok", "ok", "ok"], dtype=object),
        adaptive_filter="lms",
        motion_end_s=2.0,
        params=ProtocolTrialParams(
            tracker_mode="legacy",
            Recovery_Grace_S=2.0,
            Recovery_Diff_Bpm=10.0,
        ),
    )

    assert result.final_hr_bpm.tolist() == [70.0, 73.5, 77.0, 78.5, 80.0]
    assert result.final_source.tolist() == [
        "baseline_fft",
        "adaptive_lms",
        "adaptive_lms",
        "adaptive_lms",
        "adaptive_lms",
    ]
    assert "waiting_for_simple_crossover" in result.fusion_reason[3]
    assert result.handoff_state[4] == "adaptive_wait"


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


def test_deployment_global_uses_baseline_rest_adaptive_motion_and_fused_recovery() -> (
    None
):
    result = fuse_final_hr(
        time_s=np.asarray([0.0, 1.0, 2.0, 12.0]),
        baseline_hr_bpm=np.asarray([70.0, 70.0, 70.0, 75.0]),
        adaptive_hr_bpm=np.asarray([95.0, 100.0, 101.0, 120.0]),
        segment_label=np.asarray(
            ["rest", "motion", "motion", "recovery"], dtype=object
        ),
        qc_status=np.asarray(["ok", "ok", "ok", "ok"], dtype=object),
        adaptive_filter="lms",
        motion_end_s=2.0,
        target_scope="global",
        params=ProtocolTrialParams(
            tracker_mode="legacy",
            global_objective_strategy="deployment_global",
            Recovery_Grace_S=2.0,
            Recovery_Diff_Bpm=10.0,
        ),
    )

    assert result.final_hr_bpm.tolist() == [70.0, 73.5, 77.0, 78.5]
    assert result.final_source.tolist() == [
        "baseline_fft",
        "adaptive_lms",
        "adaptive_lms",
        "adaptive_lms",
    ]
    assert result.fusion_reason[0] == "rest_baseline"


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


def test_final_phase_dynamics_has_distinct_up_and_down_policies():
    from ppg_hr.experimental.fusion import apply_phase_dynamics

    p = ProtocolTrialParams()
    for phase, up, down in (
        ("rest", 1.5, 1.5),
        ("motion", 3.5, 1.5),
        ("recovery", 1.5, 3.0),
    ):
        assert (
            apply_phase_dynamics(np.array([100.0, 150.0]), np.array([phase, phase]), p)[
                1
            ]
            == 100 + up
        )
        assert (
            apply_phase_dynamics(np.array([100.0, 50.0]), np.array([phase, phase]), p)[
                1
            ]
            == 100 - down
        )


def test_handoff_deadline_scope_latching_and_episode_reset():
    time = np.arange(30.0)
    labels = np.array(["motion"] + ["recovery"] * 22 + ["motion"] + ["recovery"] * 6)
    baseline = np.full(30, 80.0)
    adapt = np.full(30, 120.0)
    common = dict(
        time_s=time,
        baseline_hr_bpm=baseline,
        reset_fft_hr_bpm=baseline,
        adaptive_hr_bpm=adapt,
        segment_label=labels,
        qc_status=np.full(30, "ok"),
        adaptive_filter="lms",
        motion_end_s=1.0,
        motion_runs=((0.0, 1.0), (23.0, 24.0)),
        params=ProtocolTrialParams(),
    )
    all_scope = fuse_final_hr(**common, target_scope="motion_recovery")
    assert all_scope.final_source[20] == "adaptive_lms"
    assert all_scope.handoff_reason[21] == "deadline"
    assert all_scope.final_source[22] == "reset_fft"
    assert all_scope.final_source[24] == "adaptive_lms"
    post = fuse_final_hr(**common, target_scope="motion_post10")
    assert post.handoff_reason[11] == "scope_ended"
    only = fuse_final_hr(**common, target_scope="motion_only")
    assert only.final_source[1] == "reset_fft"
    assert np.max(abs(np.diff(all_scope.final_hr_bpm))) <= 5.5


def test_handoff_requires_consecutive_valid_pairs_and_retries_invalid_reset():
    time = np.arange(25.0)
    reset = np.full(25, 101.0)
    reset[6] = np.nan
    common = dict(
        time_s=time,
        baseline_hr_bpm=np.full(25, 100.0),
        reset_fft_hr_bpm=reset,
        adaptive_hr_bpm=np.full(25, 100.0),
        segment_label=np.array(["motion"] + ["recovery"] * 24),
        qc_status=np.full(25, "ok"),
        adaptive_filter="lms",
        motion_end_s=1.0,
        params=ProtocolTrialParams(),
    )
    result = fuse_final_hr(**common)
    assert result.handoff_reason[8] == "stable_crossover"
    assert result.final_source[7] == "adaptive_lms"
    assert result.final_source[9] == "reset_fft"
    reset[21] = np.nan
    far = {**common, "adaptive_hr_bpm": np.full(25, 150.0)}
    result = fuse_final_hr(**far)
    assert result.final_source[21] == "adaptive_lms"
    assert result.handoff_reason[22] == "deadline"
