from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.preprocess.data_loader import QC_COLUMNS, SENSOR_COLUMNS, load_dataset
from ppg_hr.experimental.alignment import AlignedDataset, AlignmentInfo
from ppg_hr.experimental.cascade_solver import _TrialBase, _run_windows
from ppg_hr.experimental.preprocess_protocol import (
    ProtocolDataset,
    apply_ppg_input_transform,
    load_and_preprocess_protocol,
)
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.segmentation import SegmentInfo
from ppg_hr.params import CascadeScheme, TargetScope


def _sensor_frame(n: int = 180, *, include_time_qc: bool = True) -> pd.DataFrame:
    t = np.arange(n, dtype=float) / 100.0
    frame = pd.DataFrame(
        {
            "Uc1(mV)": 800.0 + np.sin(t),
            "Uc2(mV)": 810.0 + np.cos(t),
            "Ut1(mV)": 1600.0 + np.sin(2 * t),
            "Ut2(mV)": 1610.0 + np.cos(2 * t),
            "AccX(g)": 0.02 * np.sin(2 * np.pi * t),
            "AccY(g)": 0.01 * np.cos(2 * np.pi * t),
            "AccZ(g)": 1.0 + 0.01 * np.sin(2 * np.pi * t),
            "GyroX(dps)": 0.1 * np.sin(t),
            "GyroY(dps)": 0.1 * np.cos(t),
            "GyroZ(dps)": 0.1 * np.sin(2 * t),
            "PPG_Green": 6000.0 + 100.0 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 6500.0 + 80.0 * np.sin(2 * np.pi * 1.1 * t),
            "PPG_IR": 7000.0 + 60.0 * np.sin(2 * np.pi * 1.0 * t),
        }
    )
    if include_time_qc:
        frame.insert(0, "MissingBefore", np.where(np.arange(n) == 20, 3, 0))
        frame.insert(0, "GapLen", np.where(np.arange(n) == 20, 3, 0))
        frame.insert(0, "InterpFlag", np.where(np.arange(n) % 11 == 0, 1, 0))
        frame.insert(0, "ValidFlag", np.where(np.arange(n) < 5, 0, 1))
        frame.insert(0, "Seq", 1000 + np.arange(n))
        frame.insert(0, "SampleIndex", np.arange(n))
        frame.insert(0, "Time(s)", 12.5 + t)
    return frame


def _ref_csv(path: Path) -> Path:
    path.write_text("a,b,c\nx,y,z\n1,2,3\n0,0,72\n1,1,73\n2,2,74\n", encoding="utf-8")
    return path


def _protocol_ref_csv(path: Path) -> Path:
    pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "hr_bpm": [72.0, 73.0, 74.0]}).to_csv(path, index=False)
    return path


def test_load_dataset_preserves_new_time_and_qc_columns(tmp_path: Path) -> None:
    sensor_csv = tmp_path / "new_layout.csv"
    ref_csv = _ref_csv(tmp_path / "ref.csv")
    frame = _sensor_frame(include_time_qc=True)
    shuffled = frame[["PPG_IR", "Time(s)", "SampleIndex", *[SENSOR_COLUMNS[k] for k in SENSOR_COLUMNS if k != "PPG_IR"], "Seq", "ValidFlag", "InterpFlag", "GapLen", "MissingBefore"]]
    shuffled.to_csv(sensor_csv, index=False)

    dataset = load_dataset(sensor_csv, ref_csv)

    assert np.isclose(dataset.data["Time_s"].iloc[0], 12.5)
    for column in QC_COLUMNS:
        assert column in dataset.data.columns
    assert dataset.data["SampleIndex"].iloc[3] == 3
    assert dataset.data["ValidFlag"].iloc[0] == 0


def test_load_dataset_fills_default_qc_for_old_layout(tmp_path: Path) -> None:
    sensor_csv = tmp_path / "old_layout.csv"
    ref_csv = _ref_csv(tmp_path / "ref.csv")
    _sensor_frame(include_time_qc=False).to_csv(sensor_csv, index=False)

    dataset = load_dataset(sensor_csv, ref_csv)

    assert np.allclose(dataset.data["Time_s"].iloc[:3], [0.0, 0.01, 0.02])
    assert dataset.data["ValidFlag"].eq(1).all()
    assert dataset.data["InterpFlag"].eq(0).all()
    assert dataset.data["GapLen"].eq(0).all()
    assert dataset.data["SampleIndex"].iloc[-1] == len(dataset.data) - 1


def test_protocol_loader_carries_qc_metadata_into_frame(tmp_path: Path) -> None:
    sensor_csv = tmp_path / "multi_tiaosheng1.csv"
    ref_csv = _protocol_ref_csv(tmp_path / "multi_tiaosheng1_ref.csv")
    _sensor_frame(include_time_qc=True).to_csv(sensor_csv, index=False)

    dataset = load_and_preprocess_protocol(sensor_csv, ref_csv, fs_origin=100)
    frame = dataset.to_frame()

    for column in QC_COLUMNS:
        assert column in frame.columns
    assert np.isclose(frame["time_s"].iloc[0], 12.5)
    assert frame["ValidFlag"].iloc[0] == 0
    assert frame["InterpFlag"].sum() > 0


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
        raw_ppg_green=raw,
        raw_ppg_red=raw + 10.0,
        raw_ppg_ir=raw + 20.0,
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
        raw_ppg_green=raw,
        raw_ppg_red=raw,
        raw_ppg_ir=raw,
    )

    transformed = apply_ppg_input_transform(
        dataset,
        ProtocolTrialParams(ppg_input_transform="log_absorbance", log_absorbance_eps=1e-6),
    )

    assert transformed.ppg_green.shape == raw.shape
    assert np.all(np.isfinite(transformed.ppg_green))


def test_window_qc_stats_default_to_fallback_baseline() -> None:
    fs = 20
    n = fs * 4
    t = np.arange(n, dtype=float) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t)
    zeros = np.zeros_like(t)
    qc = pd.DataFrame(
        {
            "SampleIndex": np.arange(n),
            "Seq": np.arange(n),
            "ValidFlag": np.r_[np.zeros(fs * 2), np.ones(fs * 2)],
            "InterpFlag": np.r_[np.ones(fs * 2), np.zeros(fs * 2)],
            "GapLen": np.r_[np.full(fs * 2, 5), np.zeros(fs * 2)],
            "MissingBefore": np.r_[np.full(fs * 2, 1), np.zeros(fs * 2)],
            "raw_missing_any": np.r_[np.ones(fs * 2), np.zeros(fs * 2)],
            "raw_missing_count": np.r_[np.ones(fs * 2), np.zeros(fs * 2)],
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
    )
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=0.0,
        motion_end_s=4.0,
        motion_threshold=0.0,
        window_starts_s=np.asarray([0.0, 2.0]),
        window_centers_s=np.asarray([1.0, 3.0]),
        window_std=np.asarray([], dtype=float),
        motion_flags=np.asarray([], dtype=bool),
        labels=np.asarray(["motion", "motion"], dtype=object),
    )
    aligned = AlignedDataset(
        dataset=dataset,
        segment_info=segment,
        alignment_info=AlignmentInfo(0.0, {0.0: 0.0}, 1.0, 2, train_tw_s=2.0),
        window_starts_s=np.asarray([0.0, 2.0]),
        window_centers_s=np.asarray([1.0, 3.0]),
        segment_labels=np.asarray(["motion", "motion"], dtype=object),
        ref_hr_bpm=np.asarray([72.0, 72.0]),
        rest_indices=np.asarray([], dtype=int),
        motion_indices=np.asarray([0, 1], dtype=int),
        recovery_indices=np.asarray([], dtype=int),
    )
    base = _TrialBase(dataset=dataset, fs=fs, segment_info=segment, aligned=aligned, motion_frequency=1.2)

    frame = _run_windows(
        base,
        CascadeScheme.ACC3,
        TargetScope.MOTION_ONLY,
        ProtocolTrialParams(Fs_Target=fs, TW=2, max_order=4, K_max=2),
        1.2,
        collect_frame=True,
        collect_stages=True,
    ).frame

    assert frame.loc[0, "qc_status"] == "fallback_baseline"
    assert "ValidFlag_ratio" in frame.columns
    assert frame.loc[0, "missing_ratio"] > frame.loc[1, "missing_ratio"]
    assert frame.loc[0, "adaptive_hr_bpm"] == frame.loc[0, "baseline_ppg_hr_bpm"]
