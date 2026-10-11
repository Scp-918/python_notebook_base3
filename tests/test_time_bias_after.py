from __future__ import annotations

from ppg_hr.experimental.preprocess_protocol import SourceImu

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.experimental.alignment import search_time_bias_after
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.params import ReferenceConfig, TargetScope


def test_time_bias_after_sign_convention_samples_ref_at_pred_time_plus_bias() -> None:
    ref_time = np.arange(0, 100, dtype=float)
    ref_hr = 60.0 + 0.1 * ref_time
    pred_time = np.arange(10, 90, dtype=float)
    pred_hr = np.interp(pred_time + 3.0, ref_time, ref_hr)

    result = search_time_bias_after(
        pred_time,
        pred_hr,
        ref_time,
        ref_hr,
        search_range_s=(-5.0, 5.0),
        search_step_s=1.0,
    )

    assert result.status == "ok"
    assert result.time_bias_after_s == 3.0
    np.testing.assert_allclose(result.ref_hr_after_bpm, pred_hr)


def test_time_bias_after_scores_only_finite_overlap_and_fails_cleanly_when_none() -> None:
    ref_time = np.arange(0, 10, dtype=float)
    ref_hr = 70.0 + ref_time
    pred_time = np.arange(8, 14, dtype=float)
    pred_hr = np.interp(pred_time, ref_time, ref_hr, left=np.nan, right=np.nan)
    pred_hr[np.isnan(pred_hr)] = 99.0

    result = search_time_bias_after(
        pred_time,
        pred_hr,
        ref_time,
        ref_hr,
        search_range_s=(-2.0, 0.0),
        search_step_s=1.0,
        min_valid=2,
    )

    assert result.status == "ok"
    assert int(result.score_table.loc[result.score_table["selected"], "n_valid"].iloc[0]) >= 2

    failed = search_time_bias_after(
        np.asarray([50.0, 51.0]),
        np.asarray([80.0, 81.0]),
        ref_time,
        ref_hr,
        search_range_s=(-1.0, 1.0),
        search_step_s=1.0,
        min_valid=2,
    )
    assert failed.status == "failed"
    assert failed.time_bias_after_s == 0.0
    assert "overlapping" in failed.reason


def test_time_bias_after_does_not_mutate_prediction_array() -> None:
    ref_time = np.arange(0, 20, dtype=float)
    ref_hr = 60.0 + ref_time
    pred_time = np.arange(2, 10, dtype=float)
    pred_hr = np.interp(pred_time + 1.0, ref_time, ref_hr)
    original = pred_hr.copy()

    _ = search_time_bias_after(pred_time, pred_hr, ref_time, ref_hr)

    np.testing.assert_allclose(pred_hr, original)


def _redraw_dataset() -> ProtocolDataset:
    time_s = np.arange(6, dtype=float)
    zeros = np.zeros_like(time_s)
    return ProtocolDataset(
        sample_stem="multi_mock1",
        fs=1,
        time_s=time_s,
        ppg_green=zeros,
        ppg_red=zeros,
        ppg_ir=zeros,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=zeros,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=time_s,
        ref_hr_bpm=70.0 + time_s,
        source_imu=SourceImu(1,time_s,zeros,zeros,zeros,zeros,zeros,zeros),

        ppg_clean_green=zeros,
        ppg_clean_red=zeros,
        ppg_clean_ir=zeros,
    )


@pytest.mark.parametrize("aggregate_bias", [False, True])
def test_redraw_uses_saved_time_bias_after_and_writes_valid_global_csv(
    tmp_path: Path,
    monkeypatch,
    aggregate_bias: bool,
) -> None:
    best_csv = tmp_path / "best_params_and_alignment.csv"
    params = ProtocolTrialParams(Enable_Time_Bias_After=True, Time_Bias_After_Range_S=(-5.0, 5.0), Time_Bias_After_Step_S=1.0)
    result = rbp._ModeOptimisation(
        motion_type="write", target_scope=TargetScope.MOTION_ONLY,
        reference_config=ReferenceConfig(('ACC',)), adaptive_filter="lms", objective_mode="aae",
        data_split_mode="all_train", best_params=params, best_repeat_idx=0, best_trial_idx=0,
        n_trials=1, n_repeats=1, train_metrics={}, val_metrics={},
        test_metrics={
            "time_bias_after_s": float("nan") if aggregate_bias else 2.0,
            "time_bias_after_mode": "per_run_posthoc_oracle_alignment" if aggregate_bias else "posthoc_oracle_alignment",
        },
        per_group_rows=[], history=[], success=True, reason="",
    )
    rbp._write_motion_type_outputs(tmp_path, "write", [result])
    saved = pd.read_csv(best_csv)
    assert bool(pd.isna(saved.loc[0, "time_bias_after_s"])) is aggregate_bias

    dataset = _redraw_dataset()
    frame = pd.DataFrame(
        {
            "sample": ["multi_mock1"] * 4,
            "group_id": ["mock1"] * 4,
            "target_scope": ["motion_only"] * 4,
            "reference_order_key": ["ACC"] * 4,
            "adaptive_filter": ["lms"] * 4,
            "window_idx": [0, 1, 2, 3],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "segment_label": ["rest", "motion", "motion", "recovery"],
            "ref_hr_bpm": [70.0, 71.0, 72.0, 73.0],
            "baseline_ppg_hr_bpm": [70.0, 71.0, 72.0, 73.0],
            "adaptive_hr_bpm": [70.0, 73.0, 74.0, 73.0],
            "is_filtered_segment": [False, True, True, False],
        }
    )
    run = rbp.ProtocolRunResult(
        success=True,
        reason="",
        target_scope=TargetScope.MOTION_ONLY,
        reference_config=ReferenceConfig(('ACC',)),
        adaptive_filter="lms",
        params=ProtocolTrialParams(),
        objective_aae_bpm=1.0,
        baseline_aae_bpm=1.0,
        adaptive_aae_bpm=1.0,
        final_aae_bpm=1.0,
        baseline_acc_pct=100.0,
        adaptive_acc_pct=100.0,
        final_acc_pct=100.0,
        frame=frame,
        segment_info=None,
        alignment_info=None,
        motion_frequency=None,
        metric_arrays={},
        time_bias_after=search_time_bias_after(
            np.asarray([1.0, 2.0]), np.asarray([73.0, 74.0]),
            dataset.ref_time_s, dataset.ref_hr_bpm,
            search_range_s=(-5.0, 5.0), search_step_s=1.0,
        ),
    )

    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "run_protocol_trial", lambda *args, **kwargs: run)
    monkeypatch.setattr(rbp, "_global_combined_hr", lambda *args, **kwargs: np.asarray([np.nan, 73.0, 74.0, np.nan]))

    paths = rbp.redraw_best_param_hr_curves(
        sensor_csv_path=tmp_path / "multi_mock1.csv",
        ref_csv_path=tmp_path / "multi_mock1_ref.csv",
        target_scope="motion_only",
        reference_config="ACC",
        adaptive_filter="lms",
        best_param_csv_path=best_csv,
        output_dir=tmp_path / "redraw",
    )

    csv_path = paths["global_csv"]
    assert csv_path.exists()
    out = pd.read_csv(csv_path)
    assert {"ref_hr_after_bpm", "ppg_hr_bpm", "time_bias_after_s"}.issubset(out.columns)
    assert out["time_bias_after_s"].nunique() == 1
    assert float(out["time_bias_after_s"].iloc[0]) == 2.0
    assert np.isfinite(out["ref_hr_after_bpm"]).all()
    assert np.isfinite(out["ppg_hr_bpm"]).all()
    assert out["time_s"].tolist() == [1.0, 2.0]
    expected_source = "current_run" if aggregate_bias else "saved_best_params"
    assert out["time_bias_after_source"].eq(expected_source).all()
