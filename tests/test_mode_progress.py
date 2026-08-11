from __future__ import annotations

import json

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


def _result(*, scheme: CascadeScheme = CascadeScheme.HF2) -> rbp._ModeOptimisation:
    return rbp._ModeOptimisation(
        motion_type="write",
        target_scope=TargetScope.MOTION_POST10,
        cascade_scheme=scheme,
        adaptive_filter="lms",
        objective_mode="posthoc_aae",
        data_split_mode="all_train",
        best_params=ProtocolTrialParams(),
        best_repeat_idx=0,
        best_trial_idx=3,
        n_trials=4,
        n_repeats=1,
        train_metrics={},
        val_metrics={},
        test_metrics={
            "posthoc_final_aae_bpm": 2.25,
            "posthoc_final_acc_pct": 91.5,
            "num_windows": 12,
        },
        per_group_rows=[],
        history=[],
        success=True,
        reason="",
        training_elapsed_s=12.5,
    )


def test_acc_compare_exposes_posthoc_metrics_without_changing_legacy_fields() -> None:
    fields = rbp._acc3_compare_fields(
        {
            "final_aae_bpm": 6.0,
            "final_acc_pct": 70.0,
            "posthoc_final_aae_bpm": 3.5,
            "posthoc_final_acc_pct": 88.0,
            "num_windows": 10,
        },
        status="ok",
        reason="",
    )

    assert fields["acc3_compare_aae_bpm"] == 6.0
    assert fields["acc3_compare_accuracy_pct"] == 70.0
    assert fields["acc_compare_posthoc_aae_bpm"] == 3.5
    assert fields["acc_compare_posthoc_accuracy_pct"] == 88.0


def test_completed_mode_progress_event_contains_final_and_acc_comparison() -> None:
    result = _result()
    result.acc3_compare_metrics = rbp._acc3_compare_fields(
        {
            "final_aae_bpm": 7.0,
            "final_acc_pct": 65.0,
            "posthoc_final_aae_bpm": 4.0,
            "posthoc_final_acc_pct": 82.0,
            "num_windows": 12,
        },
        status="ok",
        reason="",
    )

    event = rbp._mode_completion_progress_payload(result, mode_idx=2, mode_total=24, resumed=False)

    assert event == {
        "stage": "optimization_mode_completed",
        "mode_idx": 2,
        "mode_current": 2,
        "mode_total": 24,
        "motion_type": "write",
        "target_scope": "MOTION_POST10",
        "target_scope_value": "motion_post10",
        "cascade_scheme": "HF2",
        "cascade_scheme_display": "HF2",
        "adaptive_filter": "lms",
        "success": True,
        "reason": "",
        "resumed": False,
        "training_elapsed_s": 12.5,
        "posthoc_final_aae_bpm": 2.25,
        "posthoc_final_acc_pct": 91.5,
        "acc_compare_posthoc_aae_bpm": 4.0,
        "acc_compare_posthoc_accuracy_pct": 82.0,
        "acc_compare_status": "ok",
        "acc_compare_reason": "",
    }


def test_acc_mode_uses_original_posthoc_metrics_for_comparison() -> None:
    result = _result(scheme=CascadeScheme.ACC3)
    metrics = rbp._acc3_compare_metrics_for(result)

    assert metrics["acc3_compare_status"] == "same_as_original"
    assert metrics["acc_compare_posthoc_aae_bpm"] == 2.25
    assert metrics["acc_compare_posthoc_accuracy_pct"] == 91.5


def test_mode_duration_and_posthoc_acc_metrics_roundtrip_manifest() -> None:
    result = _result()
    result.acc3_compare_metrics = rbp._acc3_compare_fields(
        {"posthoc_final_aae_bpm": 4.0, "posthoc_final_acc_pct": 82.0},
        status="ok",
        reason="",
    )

    payload = json.loads(json.dumps(rbp._mode_manifest_payload(result)))
    restored = rbp._mode_result_from_manifest_payload(payload)

    assert restored.training_elapsed_s == 12.5
    assert restored.acc3_compare_metrics["acc_compare_posthoc_aae_bpm"] == 4.0

