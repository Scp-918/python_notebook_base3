"""Smoke tests for unaligned PPG-HR diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ppg_hr.experimental.batch_pairing import SamplePair
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_outputs import (
    plot_raw_ppg_and_unaligned_hr_by_motion_type,
    plot_unaligned_fullfield_ppg_hr_by_motion_type,
)


def _dataset_for_plot(fs: int = 20, duration_s: float = 80.0) -> ProtocolDataset:
    """构造带 6+6 转换段的合成数据，用于测试绘图函数能正常落盘。"""

    t = np.arange(int(fs * duration_s), dtype=float) / fs
    ppg = np.sin(2.0 * np.pi * 1.2 * t)
    ref_time = np.arange(int(duration_s), dtype=float)
    ref_hr = np.full(ref_time.size, 72.0, dtype=float)
    accx = np.zeros_like(t)
    motion = (t >= 35.0) & (t < 50.0)
    accx[motion] = 0.5 * np.sin(2.0 * np.pi * 1.0 * t[motion])
    zeros = np.zeros_like(t)
    return ProtocolDataset(
        sample_stem="subject_name_write_1",
        fs=fs,
        time_s=t,
        ppg_green=ppg,
        ppg_red=0.8 * ppg,
        ppg_ir=0.6 * ppg,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=accx,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=ref_time,
        ref_hr_bpm=ref_hr,
    )


def test_plot_unaligned_fullfield_ppg_hr_by_motion_type_writes_png(tmp_path: Path) -> None:
    pair = SamplePair(
        motion_id="subject_name_write_1",
        motion_type="write",
        motion_index=1,
        stem="subject_name_write_1",
        sensor_csv=tmp_path / "subject_name_write_1_sensor.csv",
        ref_csv=tmp_path / "subject_name_write_1_HRdata.csv",
    )

    paths = plot_unaligned_fullfield_ppg_hr_by_motion_type(
        pairs=[pair],
        datasets={"subject_name_write_1": _dataset_for_plot()},
        output_dir=tmp_path / "allfield",
        fs_target=20,
        TW=8,
    )

    assert set(paths) == {"write"}
    assert paths["write"].name == "write_all_alignment_TW8.png"
    assert paths["write"].exists()
    assert paths["write"].stat().st_size > 0


def test_plot_raw_ppg_and_unaligned_hr_by_motion_type_writes_rest_dual_axis_png(tmp_path: Path) -> None:
    pair = SamplePair(
        motion_id="subject_name_write_1",
        motion_type="write",
        motion_index=1,
        stem="subject_name_write_1",
        sensor_csv=tmp_path / "missing_sensor.csv",
        ref_csv=tmp_path / "missing_ref.csv",
    )

    paths = plot_raw_ppg_and_unaligned_hr_by_motion_type(
        pairs=[pair],
        datasets={"subject_name_write_1": _dataset_for_plot()},
        output_dir=tmp_path / "allfield",
        fs_target=20,
        fs_origin=20,
        TW=8,
    )

    assert set(paths) == {"write"}
    assert paths["write"].name == "write_rest_raw_ppg_hr_dual_axis_TW8.png"
    assert paths["write"].exists()
    assert paths["write"].stat().st_size > 0
