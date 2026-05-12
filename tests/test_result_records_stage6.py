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
    assert {"best_params_json", "TW_F", "normalization_mode", "best_tdelay_s"}.issubset(params_df.columns)
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
