from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ppg_hr.experimental.alignment import AlignedDataset, AlignmentInfo
from ppg_hr.experimental.cascade_solver import (
    _TrialBase,
    _cascade_filter_window,
    _run_windows,
    clear_all_caches,
    clear_trial_heavy_caches,
)
from ppg_hr.experimental.envelope_delay import (
    ChannelDelay,
    DelayEstimate,
    NUMBA_AVAILABLE,
    _best_delay,
    _best_delay_numba,
    _best_delay_python_reference,
)
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.run_batch_protocol import _ModeOptimisation, _aggregate_logo_fold_results
from ppg_hr.experimental.segmentation import SegmentInfo
from ppg_hr.params import CascadeScheme, TargetScope


def _mode_result(fold_id: int, heldout: str, aae: float, params: ProtocolTrialParams) -> _ModeOptimisation:
    arrays = {
        "ref_hr_bpm": np.asarray([70.0, 72.0]),
        "baseline_hr_bpm": np.asarray([71.0, 73.0]),
        "adaptive_hr_bpm": np.asarray([70.0 + aae, 72.0 + aae]),
        "adaptive_abs_err_bpm": np.asarray([aae, aae]),
        "baseline_abs_err_bpm": np.asarray([1.0, 1.0]),
        "filtered_mask": np.asarray([True, True]),
    }
    return _ModeOptimisation(
        motion_type="walk",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="leave_one_group_out",
        best_params=params,
        best_repeat_idx=0,
        best_trial_idx=fold_id,
        n_trials=1,
        n_repeats=1,
        train_metrics={"success": True, "adaptive_aae_bpm": aae, "adaptive_acc_pct": 100.0},
        val_metrics={"success": False},
        test_metrics={"success": True, "adaptive_aae_bpm": aae, "adaptive_acc_pct": 100.0},
        per_group_rows=[],
        history=[],
        success=True,
        reason="",
        fold_id=fold_id,
        heldout_group_id=heldout,
        train_group_ids=["g1", "g2"],
        test_group_id=heldout,
        metric_arrays_by_split={"test": arrays, "train": arrays},
        result_level="fold",
        params_semantics="fold_best_params",
    )


def test_logo_aggregate_result_semantics_are_explicit() -> None:
    fold0 = _mode_result(0, "g0", 2.0, ProtocolTrialParams(TW=6))
    fold1 = _mode_result(1, "g1", 1.0, ProtocolTrialParams(TW=10))

    aggregate = _aggregate_logo_fold_results(
        motion_type="walk",
        scope=TargetScope.MOTION_ONLY,
        scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="leave_one_group_out",
        fold_results=[fold0, fold1],
        n_trials=1,
        n_repeats=1,
    )

    assert fold0.result_level == "fold"
    assert aggregate.result_level == "aggregate"
    assert aggregate.aggregation == "logo_window_concat"
    assert "representative" in aggregate.params_semantics
    assert "not_global" in aggregate.params_semantics
    assert aggregate.representative_fold_id == 1
    assert aggregate.representative_heldout_group_id == "g1"


def test_layered_cache_clear_preserves_global_tdelay_until_full_clear() -> None:
    marker = object()
    dataset = SimpleNamespace(_trial_base_cache={"heavy": object()}, _global_tdelay_cache={"global": marker})

    clear_trial_heavy_caches(dataset)  # type: ignore[arg-type]
    assert getattr(dataset, "_trial_base_cache") == {}
    assert getattr(dataset, "_global_tdelay_cache") == {"global": marker}

    setattr(dataset, "_trial_base_cache", {"heavy": object()})
    clear_all_caches(dataset)  # type: ignore[arg-type]
    assert getattr(dataset, "_trial_base_cache") == {}
    assert getattr(dataset, "_global_tdelay_cache") == {}


def test_best_delay_numba_matches_python_reference() -> None:
    rng = np.random.default_rng(2026)
    cases = []
    x = rng.normal(size=128)
    cases.append((x, np.roll(x, 5), 10))
    cases.append((x, np.roll(x, -7), 10))
    cases.append((rng.normal(size=97), rng.normal(size=97), 12))
    cases.append((np.ones(32), np.ones(32), 8))

    for ppg, comp, max_lag in cases:
        ref_lag, ref_corr = _best_delay_python_reference(ppg, comp, max_lag)
        got_lag, got_corr = _best_delay(ppg, comp, max_lag)
        assert got_lag == ref_lag
        assert abs(got_corr - ref_corr) < 1e-10
        if NUMBA_AVAILABLE:
            nb_lag, nb_corr = _best_delay_numba(ppg, comp, max_lag)
            assert nb_lag == ref_lag
            assert abs(nb_corr - ref_corr) < 1e-10
        else:
            pytest.skip("numba is not installed")


def _stage_dataset(fs: int = 20) -> tuple[ProtocolDataset, AlignedDataset]:
    n = fs * 8
    t = np.arange(n, dtype=float) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t)
    acc = np.sin(2 * np.pi * 1.2 * t)
    zeros = np.zeros_like(t)
    ref_time = np.arange(8, dtype=float)
    ref_hr = np.full(ref_time.size, 72.0)
    dataset = ProtocolDataset(
        sample_stem="multi_stage1",
        fs=fs,
        time_s=t,
        ppg_green=ppg,
        ppg_red=ppg,
        ppg_ir=ppg,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=acc,
        accy=0.5 * acc,
        accz=0.25 * acc,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=ref_time,
        ref_hr_bpm=ref_hr,
    )
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=0.0,
        motion_end_s=8.0,
        motion_threshold=0.0,
        window_starts_s=np.asarray([0.0, 1.0]),
        window_centers_s=np.asarray([1.0, 2.0]),
        window_std=np.asarray([], dtype=float),
        motion_flags=np.asarray([], dtype=bool),
        labels=np.asarray(["motion", "motion"], dtype=object),
    )
    info = AlignmentInfo(
        best_tdelay_s=0.0,
        std_by_delay={0.0: 0.0},
        ref_shift_s=1.0,
        num_windows=2,
        alignment_tw_s=2.0,
        train_tw_s=2.0,
        best_score=0.0,
        n_valid_score_windows=2,
    )
    aligned = AlignedDataset(
        dataset=dataset,
        segment_info=segment,
        alignment_info=info,
        window_starts_s=np.asarray([0.0, 1.0]),
        window_centers_s=np.asarray([1.0, 2.0]),
        segment_labels=np.asarray(["motion", "motion"], dtype=object),
        ref_hr_bpm=np.asarray([72.0, 72.0]),
        rest_indices=np.asarray([], dtype=int),
        motion_indices=np.asarray([0, 1], dtype=int),
        recovery_indices=np.asarray([], dtype=int),
    )
    return dataset, aligned


def test_stage_json_defaults_to_empty_but_can_be_enabled() -> None:
    dataset, aligned = _stage_dataset()
    base = _TrialBase(
        dataset=dataset,
        fs=dataset.fs,
        segment_info=aligned.segment_info,
        aligned=aligned,
        motion_frequency=1.2,
    )
    params = ProtocolTrialParams(Fs_Target=20, TW=2, max_order=4, M_base=1, K_max=2)

    no_stages = _run_windows(
        base,
        CascadeScheme.ACC3,
        TargetScope.MOTION_ONLY,
        params,
        1.2,
        collect_frame=True,
        collect_stages=False,
    ).frame
    assert set(no_stages["adaptive_stages_json"]) == {""}

    base.norm_window_cache.clear()
    base.delay_estimate_cache.clear()
    base.spectral_cache.clear()
    with_stages = _run_windows(
        base,
        CascadeScheme.ACC3,
        TargetScope.MOTION_ONLY,
        params,
        1.2,
        collect_frame=True,
        collect_stages=True,
    ).frame
    parsed = [json.loads(text) for text in with_stages["adaptive_stages_json"]]
    assert any(parsed)


def test_cascade_filter_window_records_klms_stage_parameters() -> None:
    fs = 20
    t = np.arange(fs * 3, dtype=float) / fs
    window = {
        "ppg_green": np.sin(2 * np.pi * 1.2 * t),
        "accx": np.sin(2 * np.pi * 1.2 * t),
        "accy": np.zeros_like(t),
        "accz": np.zeros_like(t),
    }
    params = ProtocolTrialParams(
        Fs_Target=fs,
        adaptive_filter="klms",
        klms_step_size=0.05,
        klms_sigma=1.0,
        klms_epsilon=0.1,
        klms_max_dictionary_size=30,
        klms_center_prune_policy="freeze_new_centers",
        klms_distance_mode="normalized",
        klms_normalized_update=True,
        klms_nlms_eps=1e-6,
        max_order=4,
        M_base=1,
        C_scale=1.0,
        K_max=2,
    )
    delay = ChannelDelay(
        channel="accx",
        sensor_type="ACC",
        D_opt_samples=0,
        D_opt_seconds=0.0,
        R_max=0.5,
        abs_corr=0.5,
    )
    cache = {
        (0, round(float(params.Kstop), 8), round(1.2, 8), params.delay_estimation_mode): DelayEstimate(
            by_channel={"accx": delay},
            order_by_type={"ACC": ["accx"]},
            primary_by_type={"ACC": "accx"},
        )
    }

    filtered, penalty_ref, stages, penalty_ref_channel = _cascade_filter_window(
        window,
        CascadeScheme.ACC3,
        params,
        fmove=1.2,
        fs=fs,
        delay_cache=cache,
        window_idx=0,
        collect_stages=True,
    )

    assert filtered.shape == window["ppg_green"].shape
    assert penalty_ref.shape == window["accx"].shape
    assert penalty_ref_channel == "accx"
    assert stages[0]["filter_type"] == "klms"
    assert stages[0]["mu"] == 0.05
    assert stages[0]["klms_step_size"] == 0.05
    assert stages[0]["sigma"] == 1.0
    assert stages[0]["epsilon"] == 0.1
    assert stages[0]["max_dictionary_size"] == 30
    assert stages[0]["center_prune_policy"] == "freeze_new_centers"
    assert stages[0]["distance_mode"] == "normalized"
    assert stages[0]["normalized_update"] is True
