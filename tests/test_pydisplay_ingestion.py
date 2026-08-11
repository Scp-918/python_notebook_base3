from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.preprocess.data_loader import (
    PYDISPLAY_SENSOR_COLUMNS,
    load_hrdata_csv,
    load_pydisplay_sensor_csv,
)


def _sensor_row(
    absolute_seq: int,
    *,
    frame_seq: int | None = None,
    sample_seq: int | None = None,
    segment_id: int = 0,
    parser_valid: object = True,
) -> dict[str, object]:
    row: dict[str, object] = {name: 1.0 for name in PYDISPLAY_SENSOR_COLUMNS}
    row.update(
        {
            "frame_seq": absolute_seq % 65536 if frame_seq is None else frame_seq,
            "absolute_seq_u64": absolute_seq,
            "segment_id": segment_id,
            "sample_seq": absolute_seq + 1 if sample_seq is None else sample_seq,
            "parser_valid": parser_valid,
        }
    )
    return row


def test_sensor_loader_drops_empty_and_invalid_rows_and_repairs_sequence(tmp_path: Path) -> None:
    path = tmp_path / "pjy_run_3_sensor.csv"
    rows = [
        _sensor_row(100, frame_seq=65535, sample_seq=1),
        _sensor_row(101, frame_seq=0, sample_seq=2),
        _sensor_row(103, frame_seq=2, sample_seq=3),
        _sensor_row(103, frame_seq=2, sample_seq=4),
        _sensor_row(102, frame_seq=1, sample_seq=5),
        _sensor_row(104, frame_seq=3, sample_seq=6, parser_valid="false"),
        {name: np.nan for name in PYDISPLAY_SENSOR_COLUMNS},
    ]
    pd.DataFrame(rows, columns=PYDISPLAY_SENSOR_COLUMNS).to_csv(path, index=False)

    loaded = load_pydisplay_sensor_csv(path, fs=100)

    assert loaded.frame["absolute_seq_u64"].tolist() == [100, 101, 102, 103]
    assert loaded.frame["Time_s"].tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert loaded.frame["ValidFlag"].tolist() == [1, 1, 0, 1]
    assert loaded.frame["InterpFlag"].tolist() == [0, 0, 1, 0]
    assert loaded.frame["MissingBefore"].tolist() == [0, 0, 0, 1]
    assert loaded.report.removed_empty_rows == 1
    assert loaded.report.removed_parser_invalid_rows == 1
    assert loaded.report.removed_duplicate_or_reordered_rows == 2
    assert loaded.report.inserted_missing_rows == 1
    assert loaded.report.frame_seq_mismatches == 0
    assert loaded.report.sample_seq_mismatches == 0


def test_sensor_loader_allows_segment_restart_and_continues_100hz_time(tmp_path: Path) -> None:
    path = tmp_path / "pjy_rope_1_sensor.csv"
    pd.DataFrame(
        [
            _sensor_row(50, frame_seq=50, sample_seq=10),
            _sensor_row(51, frame_seq=51, sample_seq=11),
            _sensor_row(2, frame_seq=2, sample_seq=1, segment_id=1),
            _sensor_row(3, frame_seq=3, sample_seq=2, segment_id=1),
        ],
        columns=PYDISPLAY_SENSOR_COLUMNS,
    ).to_csv(path, index=False)

    loaded = load_pydisplay_sensor_csv(path)

    assert loaded.frame["absolute_seq_u64"].tolist() == [50, 51, 2, 3]
    assert loaded.frame["Time_s"].tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert loaded.report.segment_transitions == 1


def test_sensor_loader_requires_exact_new_columns(tmp_path: Path) -> None:
    path = tmp_path / "pjy_write_1_sensor.csv"
    pd.DataFrame([{"PPG_G": 1, "parser_valid": True}]).to_csv(path, index=False)

    with pytest.raises(KeyError, match="absolute_seq_u64"):
        load_pydisplay_sensor_csv(path)


def test_hr_loader_uses_elapsed_seconds_and_hr_bpm_not_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "pjy_write_1_HRdata.csv"
    pd.DataFrame(
        {
            "timestamp_local": ["2099-01-01 12:34:56", "2099-01-01 12:34:57"],
            "elapsed_seconds": [0.5, 1.5],
            "hr_bpm": [70, 72],
        }
    ).to_csv(path, index=False)

    ref = load_hrdata_csv(path)

    np.testing.assert_allclose(ref, [[0.5, 70.0], [1.5, 72.0]])


@pytest.mark.parametrize("columns", [("timestamp_local", "hr_bpm"), ("elapsed_seconds", "heart_rate")])
def test_hr_loader_rejects_missing_explicit_columns(tmp_path: Path, columns: tuple[str, str]) -> None:
    path = tmp_path / "pjy_gripper_1_HRdata.csv"
    pd.DataFrame({columns[0]: [0, 1], columns[1]: [70, 71]}).to_csv(path, index=False)

    with pytest.raises(KeyError):
        load_hrdata_csv(path)


def test_hr_loader_rejects_empty_or_non_monotonic_data(tmp_path: Path) -> None:
    empty = tmp_path / "empty_HRdata.csv"
    pd.DataFrame(columns=["timestamp_local", "elapsed_seconds", "hr_bpm"]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="no usable rows"):
        load_hrdata_csv(empty)

    unordered = tmp_path / "unordered_HRdata.csv"
    pd.DataFrame({"elapsed_seconds": [1.0, 0.0], "hr_bpm": [70, 71]}).to_csv(unordered, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_hrdata_csv(unordered)
