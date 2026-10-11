from __future__ import annotations

from ppg_hr.experimental.preprocess_protocol import SourceImu

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from ppg_hr.preprocess.data_loader import (
    PYDISPLAY_SENSOR_COLUMNS,
    QC_COLUMNS,
)
from ppg_hr.experimental.alignment import SensorWindowPlan, AlignmentInfo
from ppg_hr.experimental.cascade_solver import _TrialBase, _run_windows
from ppg_hr.experimental.preprocess_protocol import (
    ProtocolDataset,
    apply_ppg_input_transform,
    load_and_preprocess_protocol,
    resample_protocol_dataset,
)
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.segmentation import SegmentInfo
from ppg_hr.params import ReferenceConfig, TargetScope


def test_protocol_loader_carries_qc_metadata_into_frame(tmp_path: Path) -> None:
    subject_dir = tmp_path / "subject_name"
    subject_dir.mkdir()
    sensor_csv = subject_dir / "subject_name_write_1_sensor.csv"
    ref_csv = subject_dir / "subject_name_write_1_HRdata.csv"
    n = 180
    t = np.arange(n, dtype=float) / 100.0
    sensor = pd.DataFrame({name: np.ones(n) for name in PYDISPLAY_SENSOR_COLUMNS})
    sensor["frame_seq"] = np.r_[np.arange(20), np.arange(21, n + 1)] % 65536
    sensor["absolute_seq_u64"] = np.r_[np.arange(20), np.arange(21, n + 1)]
    sensor["segment_id"] = 0
    sensor["sample_seq"] = np.arange(n) + 1
    sensor["parser_valid"] = True
    sensor["PPG_G"] = 6000.0 + 100.0 * np.sin(2 * np.pi * 1.2 * t)
    sensor["PPG_R"] = 6500.0 + 80.0 * np.sin(2 * np.pi * 1.1 * t)
    sensor["PPG_IR"] = 7000.0 + 60.0 * np.sin(2 * np.pi * t)
    sensor["Uh1"] = 1.0
    sensor["Uh2"] = 4.0
    sensor["Uh3"] = 6.0
    sensor["Uh4"] = 2.0
    sensor["Uc2"] = 0.5
    sensor["Uc3"] = 1.0
    sensor.loc[40, "PPG_G"] = 1e6
    sensor.to_csv(sensor_csv, index=False)
    pd.DataFrame(
        {"elapsed_seconds": [0.0, 1.0, 2.0], "hr_bpm": [72.0, 73.0, 74.0]}
    ).to_csv(ref_csv, index=False)
    c_result = np.empty((3, 1), dtype=object)
    c_result[:, 0] = [
        {"channel": 1, "c": 1.0},
        {"channel": 2, "c": 2.0},
        {"channel": 3, "c": 3.0},
    ]
    k_result = np.empty((2, 1), dtype=object)
    k_result[:, 0] = [
        {"mainChannel": 2, "k": 0.1},
        {"mainChannel": 3, "k": 0.2},
    ]
    savemat(tmp_path / "ck.mat", {"cResult": c_result, "kResult": k_result})

    dataset = load_and_preprocess_protocol(sensor_csv, ref_csv, fs_origin=100)
    frame = dataset.to_frame()

    for column in QC_COLUMNS:
        assert column in frame.columns
    assert np.isclose(frame["time_s"].iloc[0], 0.0)
    assert frame["ValidFlag"].iloc[20] == 0
    assert frame["InterpFlag"].iloc[20] == 1
    acquired = dataset.ppg_acquired_raw
    assert acquired.green[41] == 1e6
    assert np.isnan(acquired.green[20])
    assert dataset.ppg_clean_green[41] != 1e6
    assert np.isfinite(dataset.ppg_clean_green).all()
    quality = dataset.interface_quality_input
    assert quality.ppg_acquired_raw is acquired
    assert np.isnan(quality.ud_calculation[20]).all()
    original_quality_ud = quality.ud_calculation.copy()
    dataset.ud1[0] = 99.0
    np.testing.assert_allclose(quality.ud_calculation, original_quality_ud, equal_nan=True)
    resampled = resample_protocol_dataset(dataset, 25)
    assert resampled.fs == 25
    assert resampled.ppg_acquired_raw is acquired
    assert resampled.interface_quality_input is quality
    assert acquired.fs == 100 and len(acquired.green) == n + 1


def test_log_absorbance_transform_outputs_finite_ppg_with_original_length() -> None:
    fs = 50
    t = np.arange(fs * 6, dtype=float) / fs
    raw = 1000.0 + 80.0 * np.sin(2 * np.pi * 1.2 * t)
    zeros = np.zeros_like(t)
    dataset = ProtocolDataset(
        sample_stem="log_abs_positive",
        fs=fs,
        time_s=t,
        ppg_green=zeros.copy(),
        ppg_red=zeros.copy(),
        ppg_ir=zeros.copy(),
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
        ref_time_s=np.asarray([0.0, 1.0]),
        ref_hr_bpm=np.asarray([72.0, 73.0]),
        ppg_clean_green=raw,
        ppg_clean_red=raw + 10.0,
        ppg_clean_ir=raw + 20.0,
        source_imu=SourceImu(fs,t,zeros,zeros,zeros,zeros,zeros,zeros),
    )

    transformed = apply_ppg_input_transform(
        dataset,
        ProtocolTrialParams(ppg_input_transform="log_absorbance"),
    )

    assert transformed.ppg_green.shape == raw.shape
    assert np.all(np.isfinite(transformed.ppg_green))
    assert np.all(np.isfinite(transformed.ppg_red))
    assert np.all(np.isfinite(transformed.ppg_ir))


def test_log_absorbance_transform_handles_zero_and_negative_ppg() -> None:
    fs = 50
    t = np.arange(fs * 6, dtype=float) / fs
    raw = -20.0 + 10.0 * np.sin(2 * np.pi * 1.2 * t)
    zeros = np.zeros_like(t)
    dataset = ProtocolDataset(
        sample_stem="log_abs_negative",
        fs=fs,
        time_s=t,
        ppg_green=raw.copy(),
        ppg_red=raw.copy(),
        ppg_ir=raw.copy(),
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
        ref_time_s=np.asarray([0.0, 1.0]),
        ref_hr_bpm=np.asarray([72.0, 73.0]),
        ppg_clean_green=raw,
        ppg_clean_red=raw,
        ppg_clean_ir=raw,
        source_imu=SourceImu(fs,t,zeros,zeros,zeros,zeros,zeros,zeros),
    )

    transformed = apply_ppg_input_transform(
        dataset,
        ProtocolTrialParams(ppg_input_transform="log_absorbance", log_absorbance_eps=1e-6),
    )

    assert transformed.ppg_green.shape == raw.shape
    assert np.all(np.isfinite(transformed.ppg_green))


def test_window_qc_stats_default_to_fallback_baseline() -> None:
    fs = 20
    n = fs * 12
    t = np.arange(n, dtype=float) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t)
    zeros = np.zeros_like(t)
    qc = pd.DataFrame(
        {
            "SampleIndex": np.arange(n),
            "Seq": np.arange(n),
            "ValidFlag": np.r_[np.zeros(fs * 3), np.ones(fs * 9)],
            "InterpFlag": np.r_[np.ones(fs * 3), np.zeros(fs * 9)],
            "GapLen": np.r_[np.full(fs * 3, 5), np.zeros(fs * 9)],
            "MissingBefore": np.r_[np.full(fs * 3, 1), np.zeros(fs * 9)],
            "raw_missing_any": np.r_[np.ones(fs * 3), np.zeros(fs * 9)],
            "raw_missing_count": np.r_[np.ones(fs * 3), np.zeros(fs * 9)],
            "ud_denominator_near_zero": np.zeros(n),
        }
    )
    dataset = ProtocolDataset(
        sample_stem="multi_qc1",
        fs=fs,
        time_s=t,
        ppg_green=ppg,
        ppg_red=ppg,
        ppg_ir=ppg,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=ppg,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=np.asarray([0.0, 1.0, 2.0, 3.0]),
        ref_hr_bpm=np.full(4, 72.0),
        qc=qc,
        source_imu=SourceImu(fs,t,ppg,zeros,zeros,zeros,zeros,zeros),

        ppg_clean_green=ppg,
        ppg_clean_red=ppg,
        ppg_clean_ir=ppg,
    )
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=0.0,
        motion_end_s=12.0,
        motion_threshold=0.0,
        window_starts_s=np.asarray([0.0, 4.0]),
        window_centers_s=np.asarray([4.0, 8.0]),
        window_std=np.asarray([], dtype=float),
        motion_flags=np.asarray([], dtype=bool),
        labels=np.asarray(["motion", "motion"], dtype=object),
        motion_runs=((0.0,12.0),),
    )
    aligned = SensorWindowPlan(
        dataset=dataset,
        segment_info=segment,
        alignment_info=AlignmentInfo(0.0, {0.0: 0.0}, 1.0, 2, hr_window_s=8),
        window_starts_s=np.asarray([0.0, 4.0]),
        window_centers_s=np.asarray([4.0, 8.0]),
        segment_labels=np.asarray(["motion", "motion"], dtype=object),
        rest_indices=np.asarray([], dtype=int),
        motion_indices=np.asarray([0, 1], dtype=int),
        recovery_indices=np.asarray([], dtype=int),
    )
    base = _TrialBase(dataset=dataset, fs=fs, segment_info=segment, aligned=aligned, motion_frequency=1.2)

    frame = _run_windows(
        base,
        ReferenceConfig(('ACC',)),
        TargetScope.MOTION_ONLY,
        ProtocolTrialParams(Fs_Target=fs, hr_window_s=8, memory_ms=4, K_max=2),
        1.2,
        collect_frame=True,
        collect_stages=True,
    ).frame

    assert frame.loc[0, "qc_status"] == "fallback_baseline"
    assert "ValidFlag_ratio" in frame.columns
    assert frame.loc[0, "missing_ratio"] > frame.loc[1, "missing_ratio"]
    assert frame.loc[0, "adaptive_hr_bpm"] == frame.loc[0, "baseline_ppg_hr_bpm"]
