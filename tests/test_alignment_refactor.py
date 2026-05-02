"""Tests for decoupled global Tdelay alignment and tracked rest PPG-HR."""

from __future__ import annotations

import numpy as np

from ppg_hr.experimental.alignment import (
    align_ppg_to_ref_hr,
    apply_rest_hr_slew_limit,
    build_aligned_training_windows,
    estimate_global_tdelay_from_rest,
    extract_rest_ppg_hr_tracked,
    smooth_rest_hr_sequence,
)
from ppg_hr.experimental.cascade_solver import _global_tdelay_cache_key
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.segmentation import SegmentInfo


def _synthetic_dataset(fs: int = 50, duration_s: float = 90.0) -> tuple[ProtocolDataset, SegmentInfo]:
    """构造一个小型协议数据集：PPG 为稳定心率，ACC 仅用于占位。"""

    t = np.arange(int(fs * duration_s), dtype=float) / float(fs)
    hr_bpm = 72.0 + 3.0 * np.sin(2.0 * np.pi * t / 60.0)
    phase = np.cumsum(hr_bpm / 60.0) / float(fs)
    ppg = np.sin(2.0 * np.pi * phase)
    ref_time = np.arange(int(duration_s), dtype=float)
    ref_hr = 72.0 + 3.0 * np.sin(2.0 * np.pi * ref_time / 60.0)
    zeros = np.zeros_like(t)
    ds = ProtocolDataset(
        sample_stem="synthetic_alignment",
        fs=fs,
        time_s=t,
        ppg_green=ppg,
        ppg_red=0.8 * ppg,
        ppg_ir=0.6 * ppg,
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
        ref_time_s=ref_time,
        ref_hr_bpm=ref_hr,
    )
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=50.0,
        motion_end_s=65.0,
        motion_threshold=0.0,
        window_starts_s=np.asarray([], dtype=float),
        window_centers_s=np.asarray([], dtype=float),
        window_std=np.asarray([], dtype=float),
        motion_flags=np.asarray([], dtype=bool),
        labels=np.asarray([], dtype=str),
    )
    return ds, segment


def test_alignment_tdelay_uses_alignment_tw_while_training_windows_use_train_tw() -> None:
    ds, segment = _synthetic_dataset()
    estimate = estimate_global_tdelay_from_rest(
        ds,
        segment,
        ds.fs,
        alignment_TW=8.0,
        alignment_step_s=1.0,
        delay_range_s=(-1.0, 1.0),
        delay_step_s=0.5,
    )

    aligned_6 = build_aligned_training_windows(ds, segment, ds.fs, train_TW=6.0, best_tdelay_s=estimate.best_tdelay_s, tdelay_result=estimate)
    aligned_10 = build_aligned_training_windows(ds, segment, ds.fs, train_TW=10.0, best_tdelay_s=estimate.best_tdelay_s, tdelay_result=estimate)

    assert aligned_6.alignment_info.best_tdelay_s == aligned_10.alignment_info.best_tdelay_s
    assert aligned_6.alignment_info.alignment_tw_s == 8.0
    assert aligned_10.alignment_info.alignment_tw_s == 8.0
    assert aligned_6.alignment_info.train_tw_s == 6.0
    assert aligned_10.alignment_info.train_tw_s == 10.0
    assert aligned_6.window_centers_s.size != aligned_10.window_centers_s.size
    assert not np.array_equal(aligned_6.window_centers_s, aligned_10.window_centers_s)


def test_compat_wrapper_keeps_alignment_tw_independent_from_tw() -> None:
    ds, segment = _synthetic_dataset()
    aligned_6 = align_ppg_to_ref_hr(ds, segment, TW=6.0, fs_target=ds.fs, alignment_TW=8.0, delay_range_s=(-1.0, 1.0), delay_step_s=0.5)
    aligned_10 = align_ppg_to_ref_hr(ds, segment, TW=10.0, fs_target=ds.fs, alignment_TW=8.0, delay_range_s=(-1.0, 1.0), delay_step_s=0.5)

    assert aligned_6.alignment_info.best_tdelay_s == aligned_10.alignment_info.best_tdelay_s
    assert aligned_6.alignment_info.alignment_tw_s == 8.0
    assert aligned_10.alignment_info.alignment_tw_s == 8.0
    assert aligned_6.alignment_info.train_tw_s == 6.0
    assert aligned_10.alignment_info.train_tw_s == 10.0


def test_global_tdelay_cache_key_excludes_train_tw() -> None:
    ds, _ = _synthetic_dataset()
    p6 = ProtocolTrialParams(Fs_Target=50, TW=6.0, Alignment_TW=8.0)
    p10 = ProtocolTrialParams(Fs_Target=50, TW=10.0, Alignment_TW=8.0)
    assert _global_tdelay_cache_key(ds, p6) == _global_tdelay_cache_key(ds, p10)
    changed_peak_rule = ProtocolTrialParams(Fs_Target=50, TW=6.0, Alignment_TW=8.0, Rest_HR_Peak_Percent=0.5)
    assert _global_tdelay_cache_key(ds, p6) != _global_tdelay_cache_key(ds, changed_peak_rule)


def test_rest_hr_tracking_and_slew_limit_suppress_jump_peak() -> None:
    fs = 100
    tw_s = 8.0
    t = np.arange(int(fs * tw_s), dtype=float) / fs
    rest_ppg = np.concatenate(
        [
            np.sin(2.0 * np.pi * 1.2 * t),
            np.sin(2.0 * np.pi * 2.5 * t),
            np.sin(2.0 * np.pi * 1.2 * t),
        ]
    )
    result = extract_rest_ppg_hr_tracked(
        rest_ppg,
        fs,
        tw_s=tw_s,
        step_s=tw_s,
        hr_band_bpm=(40.0, 180.0),
        track_band_bpm=30.0,
        slew_limit_bpm=6.0,
        slew_step_bpm=4.0,
        smooth_method="none",
    )

    assert result.hr_bpm_raw.size == 3
    assert result.hr_bpm_raw[1] > 130.0
    assert result.hr_bpm_tracked[1] <= result.hr_bpm_tracked[0] + 4.1

    limited = apply_rest_hr_slew_limit(np.asarray([72.0, 150.0]), slew_limit_bpm=6.0, slew_step_bpm=4.0)
    np.testing.assert_allclose(limited, np.asarray([72.0, 76.0]))


def test_rest_hr_reference_style_penalty_is_optional_and_recorded() -> None:
    fs = 100
    tw_s = 8.0
    t = np.arange(int(fs * tw_s), dtype=float) / fs
    ppg = np.sin(2.0 * np.pi * 1.2 * t)
    motion_ref = np.sin(2.0 * np.pi * 1.2 * t)

    no_penalty = extract_rest_ppg_hr_tracked(
        ppg,
        fs,
        tw_s=tw_s,
        step_s=tw_s,
        hr_band_bpm=(40.0, 120.0),
        smooth_method="none",
        return_debug=True,
    )
    with_penalty = extract_rest_ppg_hr_tracked(
        ppg,
        fs,
        tw_s=tw_s,
        step_s=tw_s,
        hr_band_bpm=(40.0, 120.0),
        smooth_method="none",
        penalty_signal=motion_ref,
        spec_penalty_enable=True,
        return_debug=True,
    )

    assert no_penalty.quality is not None
    assert with_penalty.quality is not None
    assert not bool(no_penalty.quality["penalty_applied"][0])
    assert bool(with_penalty.quality["penalty_applied"][0])
    assert np.isfinite(with_penalty.quality["penalty_freq_hz"][0])


def test_rest_hr_moving_median_smoothing_handles_spikes_and_short_arrays() -> None:
    smoothed = smooth_rest_hr_sequence(np.asarray([70.0, 70.0, 120.0, 70.0, 70.0]), smooth_win=3)
    assert smoothed[2] == 70.0

    short = np.asarray([70.0, 120.0])
    np.testing.assert_array_equal(smooth_rest_hr_sequence(short, smooth_win=3), short)
