from __future__ import annotations

import json

import numpy as np

from ppg_hr.experimental.alignment import AlignedDataset, AlignmentInfo
from ppg_hr.experimental.cascade_solver import _TrialBase, _run_windows
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.segmentation import SegmentInfo
from ppg_hr.params import CascadeScheme, TargetScope


def _twf_base(fs: int = 20) -> _TrialBase:
    n = fs * 5
    t = np.arange(n, dtype=float) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t)
    acc = np.sin(2 * np.pi * 1.2 * t + 0.1)
    zeros = np.zeros_like(t)
    dataset = ProtocolDataset(
        sample_stem="multi_twf1",
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
        ref_time_s=np.arange(5, dtype=float),
        ref_hr_bpm=np.full(5, 72.0),
    )
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=0.0,
        motion_end_s=5.0,
        motion_threshold=0.0,
        window_starts_s=np.asarray([0.0, 1.0, 2.0]),
        window_centers_s=np.asarray([1.0, 2.0, 3.0]),
        window_std=np.asarray([], dtype=float),
        motion_flags=np.asarray([], dtype=bool),
        labels=np.asarray(["motion", "motion", "motion"], dtype=object),
    )
    aligned = AlignedDataset(
        dataset=dataset,
        segment_info=segment,
        alignment_info=AlignmentInfo(0.0, {0.0: 0.0}, 1.0, 3, train_tw_s=2.0),
        window_starts_s=np.asarray([0.0, 1.0, 2.0]),
        window_centers_s=np.asarray([1.0, 2.0, 3.0]),
        segment_labels=np.asarray(["motion", "motion", "motion"], dtype=object),
        ref_hr_bpm=np.asarray([72.0, 72.0, 72.0]),
        rest_indices=np.asarray([], dtype=int),
        motion_indices=np.asarray([0, 1, 2], dtype=int),
        recovery_indices=np.asarray([], dtype=int),
    )
    return _TrialBase(dataset=dataset, fs=fs, segment_info=segment, aligned=aligned, motion_frequency=1.2)


def test_tw_f_context_keeps_fft_center_and_records_context_lengths() -> None:
    base = _twf_base()
    params = ProtocolTrialParams(Fs_Target=20, TW=2, TW_F=1.0, max_order=4, M_base=1, K_max=2)

    frame = _run_windows(
        base,
        CascadeScheme.ACC3,
        TargetScope.MOTION_ONLY,
        params,
        1.2,
        collect_frame=True,
        collect_stages=True,
    ).frame

    np.testing.assert_allclose(frame["time_s"].to_numpy(dtype=float), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame["fft_start_s"].to_numpy(dtype=float), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(frame["fft_end_s"].to_numpy(dtype=float), [2.0, 3.0, 4.0])
    assert frame["fft_input_samples"].tolist() == [40, 40, 40]
    assert frame["adaptive_input_samples"].tolist() == [60, 60, 60]
    assert frame.loc[0, "tw_f_context_status"] == "padded_left"
    assert frame.loc[1, "tw_f_context_status"] == "full"
    assert frame.loc[0, "adaptive_start_s"] == -1.0
    assert frame.loc[0, "adaptive_source_start_s"] == 0.0

    stages = json.loads(frame.loc[1, "adaptive_stages_json"])
    assert stages
    assert "reference_channel_ranking" in stages[0]
    assert "penalty_ref_channel" in stages[0]
    assert stages[0]["adaptive_input_samples"] == 60


def test_tw_f_zero_uses_legacy_window_lengths() -> None:
    base = _twf_base()
    params = ProtocolTrialParams(Fs_Target=20, TW=2, TW_F=0.0, max_order=4, M_base=1, K_max=2)

    frame = _run_windows(
        base,
        CascadeScheme.ACC3,
        TargetScope.MOTION_ONLY,
        params,
        1.2,
        collect_frame=True,
        collect_stages=False,
    ).frame

    assert frame["fft_input_samples"].tolist() == [40, 40, 40]
    assert frame["adaptive_input_samples"].tolist() == [40, 40, 40]
    np.testing.assert_allclose(frame["adaptive_start_s"], frame["fft_start_s"])
    assert set(frame["tw_f_context_status"]) == {"full"}
