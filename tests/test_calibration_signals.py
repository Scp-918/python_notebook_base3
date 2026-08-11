from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from ppg_hr.experimental.preprocess_protocol import (
    load_and_preprocess_protocol,
    load_protocol_raw_clean_frames,
)
from ppg_hr.preprocess.calibration import CalibrationCoefficients, load_subject_calibration
from ppg_hr.preprocess.data_loader import PYDISPLAY_SENSOR_COLUMNS


def _write_calibration(
    root: Path,
    *,
    c2: float = 2.0,
    c3: float = 3.0,
    k2: float = 0.1,
    k3: float = 0.2,
    c2_channel: int = 2,
    k3_channel: int = 3,
) -> Path:
    c_result = np.empty((4, 1), dtype=object)
    for idx in range(4):
        c_result[idx, 0] = {"channel": idx + 1, "c": float(idx + 1)}
    c_result[1, 0] = {"channel": c2_channel, "c": c2}
    c_result[2, 0] = {"channel": 3, "c": c3}
    k_result = np.empty((2, 1), dtype=object)
    k_result[0, 0] = {"mainChannel": 2, "k": k2}
    k_result[1, 0] = {"mainChannel": k3_channel, "k": k3}
    path = root / "ck.mat"
    savemat(path, {"cResult": c_result, "kResult": k_result})
    return path


def _write_sensor(path: Path, n: int = 240, *, near_zero_first: bool = False) -> None:
    t = np.arange(n, dtype=float) / 100.0
    frame = pd.DataFrame({name: np.ones(n, dtype=float) for name in PYDISPLAY_SENSOR_COLUMNS})
    frame["frame_seq"] = np.arange(n) % 65536
    frame["absolute_seq_u64"] = np.arange(n)
    frame["segment_id"] = 0
    frame["sample_seq"] = np.arange(n) + 1
    frame["parser_valid"] = True
    frame["PPG_G"] = 5000.0 + 100.0 * np.sin(2 * np.pi * 1.2 * t)
    frame["PPG_R"] = 5200.0 + 80.0 * np.sin(2 * np.pi * 1.1 * t)
    frame["PPG_IR"] = 5500.0 + 60.0 * np.sin(2 * np.pi * t)
    frame["Uh1"] = 1.0 + 0.01 * np.sin(t)
    frame["Uh2"] = 4.0 + 0.02 * np.sin(t)
    frame["Uh3"] = 6.0 + 0.03 * np.cos(t)
    frame["Uh4"] = 2.0 + 0.01 * np.cos(t)
    frame["Uc2"] = 0.5
    frame["Uc3"] = 1.0
    if near_zero_first:
        frame.loc[0, "Uc2"] = 2.0
        frame.loc[0, "Uc3"] = 3.0
    frame["ACC_X"] = np.sin(t)
    frame["ACC_Y"] = np.cos(t)
    frame["ACC_Z"] = 1.0
    frame.to_csv(path, index=False)


def _write_hr(path: Path) -> None:
    pd.DataFrame(
        {"timestamp_local": ["x", "y", "z"], "elapsed_seconds": [0.0, 1.0, 2.0], "hr_bpm": [70, 71, 72]}
    ).to_csv(path, index=False)


def test_load_subject_calibration_reads_required_matlab_cells(tmp_path: Path) -> None:
    subject = tmp_path / "pjy"
    subject.mkdir()
    expected_path = _write_calibration(tmp_path)
    logs: list[str] = []

    calibration = load_subject_calibration(subject, on_log=logs.append)

    assert calibration == CalibrationCoefficients(
        source_path=expected_path.resolve(),
        source_sha256=calibration.source_sha256,
        c2=2.0,
        c3=3.0,
        k2=0.1,
        k3=0.2,
    )
    assert len(calibration.source_sha256) == 64
    assert str(expected_path.resolve()) in logs[0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"c2_channel": 9}, "cResult{2}.channel"),
        ({"k3_channel": 9}, "kResult{2}.mainChannel"),
        ({"c2": float("nan")}, "cResult{2}.c"),
    ],
)
def test_load_subject_calibration_rejects_invalid_fields(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    subject = tmp_path / "pjy"
    subject.mkdir()
    _write_calibration(tmp_path, **kwargs)

    with pytest.raises(ValueError, match=message.replace("{", r"\{").replace("}", r"\}")):
        load_subject_calibration(subject)


def test_load_subject_calibration_reports_missing_file(tmp_path: Path) -> None:
    subject = tmp_path / "pjy"
    subject.mkdir()
    with pytest.raises(FileNotFoundError, match="ck.mat"):
        load_subject_calibration(subject)


@pytest.mark.parametrize(
    ("bad_entry", "message"),
    [
        ({"channel": 2}, "cResult{2}.c"),
        ({"channel": 2, "c": np.array([1.0, 2.0])}, "cResult{2}.c must be scalar"),
    ],
)
def test_load_subject_calibration_rejects_missing_and_nonscalar_coefficients(
    tmp_path: Path,
    bad_entry: dict[str, object],
    message: str,
) -> None:
    subject = tmp_path / "pjy"
    subject.mkdir()
    _write_calibration(tmp_path)
    c_result = np.empty((3, 1), dtype=object)
    c_result[:, 0] = [
        {"channel": 1, "c": 1.0},
        bad_entry,
        {"channel": 3, "c": 3.0},
    ]
    k_result = np.empty((2, 1), dtype=object)
    k_result[:, 0] = [
        {"mainChannel": 2, "k": 0.1},
        {"mainChannel": 3, "k": 0.2},
    ]
    savemat(tmp_path / "ck.mat", {"cResult": c_result, "kResult": k_result})

    with pytest.raises(ValueError, match=message.replace("{", r"\{").replace("}", r"\}")):
        load_subject_calibration(subject)


def test_new_voltage_formulas_and_near_zero_denominator_qc(tmp_path: Path) -> None:
    subject = tmp_path / "pjy"
    subject.mkdir()
    _write_calibration(tmp_path)
    sensor = subject / "pjy_write_1_sensor.csv"
    ref = subject / "pjy_write_1_HRdata.csv"
    _write_sensor(sensor, near_zero_first=True)
    _write_hr(ref)
    calibration = load_subject_calibration(subject)

    raw, clean = load_protocol_raw_clean_frames(sensor, calibration=calibration)

    assert raw.loc[1, "cf1"] == pytest.approx(raw.loc[1, "Uh1"] * 1000.0)
    assert raw.loc[1, "cf2"] == pytest.approx(raw.loc[1, "Uh4"] * 1000.0)
    assert raw.loc[1, "hf1"] == pytest.approx((raw.loc[1, "Uh2"] - 0.1 * raw.loc[1, "Uh1"]) * 1000.0)
    assert raw.loc[1, "hf2"] == pytest.approx((raw.loc[1, "Uh3"] - 0.2 * raw.loc[1, "Uh4"]) * 1000.0)
    assert raw.loc[1, "hfcomp1"] == pytest.approx(raw.loc[1, "Uh2"] * 1000.0)
    assert raw.loc[1, "hfcomp2"] == pytest.approx(raw.loc[1, "Uh3"] * 1000.0)
    assert raw.loc[1, "ud1"] == pytest.approx((raw.loc[1, "Uh2"] - 0.5) / (2.0 - 0.5))
    assert raw.loc[1, "ud2"] == pytest.approx((raw.loc[1, "Uh3"] - 1.0) / (3.0 - 1.0))
    assert np.isnan(raw.loc[0, "ud1"])
    assert np.isnan(raw.loc[0, "ud2"])
    assert clean.loc[0, "ud_denominator_near_zero"] == 1
    assert np.isfinite(clean[["ud1", "ud2"]].to_numpy()).all()

    dataset = load_and_preprocess_protocol(sensor, ref, calibration=calibration)
    assert dataset.source_metadata["calibration_path"] == str(calibration.source_path)
    assert dataset.hfcomp1 is not None
    assert dataset.ud1 is not None
    assert len(dataset.time_s) == 240
