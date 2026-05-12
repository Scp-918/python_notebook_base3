from __future__ import annotations

from pathlib import Path

import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


def _mode_result_for_summary() -> rbp._ModeOptimisation:
    return rbp._ModeOptimisation(
        motion_type="walk",
        target_scope=TargetScope.GLOBAL,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="all_train",
        best_params=ProtocolTrialParams(),
        best_repeat_idx=0,
        best_trial_idx=0,
        n_trials=1,
        n_repeats=1,
        train_metrics={"success": True},
        val_metrics={"success": True},
        test_metrics={
            "success": True,
            "reason": "",
            "baseline_aae_bpm": 10.0,
            "adaptive_aae_bpm": 5.0,
            "final_aae_bpm": 3.0,
            "baseline_acc_pct": 20.0,
            "adaptive_acc_pct": 50.0,
            "final_acc_pct": 80.0,
            "posthoc_baseline_aae_bpm": 9.0,
            "posthoc_adaptive_aae_bpm": 4.0,
            "posthoc_final_aae_bpm": 2.0,
            "posthoc_baseline_acc_pct": 30.0,
            "posthoc_adaptive_acc_pct": 60.0,
            "posthoc_final_acc_pct": 90.0,
        },
        per_group_rows=[],
        history=[],
        success=True,
        reason="",
        result_level="single_split",
        params_semantics="best_params_for_this_split",
    )


def test_batch_objective_uses_final_metric_by_default() -> None:
    assert rbp._objective_value({"adaptive_aae_bpm": 1.0, "final_aae_bpm": 7.0}, "aae", 999.0) == 7.0
    assert rbp._objective_value({"adaptive_acc_pct": 100.0, "final_acc_pct": 40.0}, "accuracy", 999.0) == 60.0


def test_final_summary_records_baseline_adaptive_and_final_values(tmp_path: Path) -> None:
    paths = rbp._write_final_summary(
        tmp_path,
        {"walk": [_mode_result_for_summary()]},
        [TargetScope.GLOBAL],
        objective_mode="aae",
        data_split_mode="all_train",
    )

    df = pd.read_csv(paths["global_aae"])
    assert {"baseline_metric_value", "adaptive_metric_value", "final_metric_value"}.issubset(df.columns)
    assert float(df.loc[0, "baseline_metric_value"]) == 10.0
    assert float(df.loc[0, "adaptive_metric_value"]) == 5.0
    assert float(df.loc[0, "final_metric_value"]) == 3.0
    assert float(df.loc[0, "metric_value"]) == 3.0

    posthoc = pd.read_csv(paths["global_posthoc_accuracy"])
    assert float(posthoc.loc[0, "baseline_metric_value"]) == 30.0
    assert float(posthoc.loc[0, "adaptive_metric_value"]) == 60.0
    assert float(posthoc.loc[0, "final_metric_value"]) == 90.0
