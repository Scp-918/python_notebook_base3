from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


def _stage6_result() -> rbp._ModeOptimisation:
    params = ProtocolTrialParams(TW=8, TW_F=1.5, Fs_Target=50, adaptive_filter="lms")
    return rbp._ModeOptimisation(
        motion_type="tiaosheng",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="accuracy",
        data_split_mode="all_train",
        best_params=params,
        best_repeat_idx=0,
        best_trial_idx=0,
        n_trials=1,
        n_repeats=1,
        train_metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        val_metrics={"success": True},
        test_metrics={
            "success": True,
            "reason": "",
            "baseline_aae_bpm": 9.0,
            "adaptive_aae_bpm": 5.0,
            "final_aae_bpm": 3.0,
            "baseline_acc_pct": 40.0,
            "adaptive_acc_pct": 70.0,
            "final_acc_pct": 90.0,
            "posthoc_baseline_aae_bpm": 8.0,
            "posthoc_adaptive_aae_bpm": 4.0,
            "posthoc_final_aae_bpm": 2.0,
            "posthoc_baseline_acc_pct": 50.0,
            "posthoc_adaptive_acc_pct": 80.0,
            "posthoc_final_acc_pct": 95.0,
            "time_bias_after_s": 1.0,
            "num_windows": 12,
        },
        per_group_rows=[],
        history=[],
        success=True,
        reason="",
        result_level="single_split",
        params_semantics="best_params_for_this_split",
    )


def test_motion_type_outputs_include_stage6_record_files(tmp_path: Path) -> None:
    rbp._write_motion_type_outputs(tmp_path, "tiaosheng", [_stage6_result()])

    expected = [
        "best_params_and_alignment.csv",
        "best_metrics.csv",
        "motion_frequency_and_params.csv",
        "full_report.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists()

    params_df = pd.read_csv(tmp_path / "best_params_and_alignment.csv")
    assert {
        "best_params_json",
        "TW_F",
        "normalization_mode",
        "best_tdelay_s",
        "param_TW_F",
        "param_ppg_input_transform",
        "param_global_objective_strategy",
        "param_cascade_guard_policy",
    }.issubset(params_df.columns)
    assert json.loads(params_df.loc[0, "best_params_json"])["TW_F"] == 1.5

    metrics_df = pd.read_csv(tmp_path / "best_metrics.csv")
    assert {"baseline_aae_bpm", "adaptive_aae_bpm", "final_aae_bpm", "n_recovery_fallback"}.issubset(
        metrics_df.columns
    )
    assert float(metrics_df.loc[0, "final_aae_bpm"]) == 3.0

    freq_df = pd.read_csv(tmp_path / "motion_frequency_and_params.csv")
    assert {"motion_frequency_hz", "penalty_ref_channel", "reference_channel_ranking_summary"}.issubset(
        freq_df.columns
    )

    report = json.loads((tmp_path / "full_report.json").read_text(encoding="utf-8"))
    assert report["motion_type"] == "tiaosheng"
    assert "fusion_source_distribution" in report
    assert "git_commit_hash" in report


def test_flatten_params_for_record_roundtrips_all_protocol_fields() -> None:
    params = ProtocolTrialParams(
        TW_F=2.5,
        ppg_input_transform="log_absorbance",
        log_absorbance_ratio_clip=(1e-4, 1e4),
        rff_err_clip=None,
        klms_normalized_update=True,
    )

    flat = rbp.flatten_params_for_record(params)
    restored = rbp.protocol_params_from_record(flat)

    for field in params.to_dict():
        assert f"param_{field}" in flat
    assert restored.TW_F == 2.5
    assert restored.ppg_input_transform == "log_absorbance"
    assert restored.log_absorbance_ratio_clip == (1e-4, 1e4)
    assert restored.rff_err_clip is None
    assert restored.klms_normalized_update is True


def test_protocol_params_from_record_prefers_param_columns_over_json() -> None:
    json_params = ProtocolTrialParams(ppg_input_transform="raw_bandpass", TW=6).to_dict()
    row = {
        "best_params_json": json.dumps(json_params),
        "param_ppg_input_transform": "log_absorbance",
        "param_TW": 10,
        "adaptive_filter": "klms",
    }

    restored = rbp.protocol_params_from_record(row)

    assert restored.ppg_input_transform == "log_absorbance"
    assert restored.TW == 10
    assert restored.adaptive_filter == "klms"


def test_protocol_params_from_old_rff_record_uses_fixed_sigma_fallback() -> None:
    old_params = ProtocolTrialParams(adaptive_filter="rff_lms", rff_sigma=2.5).to_dict()
    old_params.pop("rff_sigma_scale", None)
    row = {"best_params_json": json.dumps(old_params)}

    restored = rbp.protocol_params_from_record(row)

    assert restored.adaptive_filter == "rff_lms"
    assert restored.rff_sigma == 2.5
    assert restored.rff_sigma_scale is None


def test_append_history_includes_params_json_and_param_columns() -> None:
    history: list[dict[str, object]] = []
    params = ProtocolTrialParams(ppg_input_transform="log_absorbance", cascade_guard_policy="rms_guard")

    rbp._append_history(
        history,
        "tiaosheng",
        TargetScope.GLOBAL,
        CascadeScheme.ACC3,
        "lms",
        params,
        repeat_idx=0,
        trial_idx=1,
        objective_value=3.0,
        metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        best_so_far=3.0,
    )

    row = history[0]
    assert isinstance(row["params"], str)
    assert json.loads(row["params"])["ppg_input_transform"] == "log_absorbance"
    assert row["param_ppg_input_transform"] == "log_absorbance"
    assert row["param_cascade_guard_policy"] == "rms_guard"


def test_best_params_all_json_can_be_rebuilt_from_saved_history_paths(tmp_path: Path) -> None:
    result = _stage6_result()
    rbp._append_history(
        result.history,
        "tiaosheng",
        TargetScope.MOTION_ONLY,
        CascadeScheme.ACC3,
        "lms",
        result.best_params,
        repeat_idx=0,
        trial_idx=0,
        objective_value=3.0,
        metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        best_so_far=3.0,
    )
    rbp._persist_mode_artifacts(tmp_path, result, checkpoint_status="done")
    result.history = []

    rbp._write_motion_type_outputs(tmp_path, "tiaosheng", [result], write_best_params_all=True)

    payload = json.loads((tmp_path / "best_params_all.json").read_text(encoding="utf-8"))
    assert payload[result.mode_key]["trial_history"][0]["trial_idx"] == 0
