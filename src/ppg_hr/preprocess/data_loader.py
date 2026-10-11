"""Sensor + ground-truth CSV ingestion (Python port of ``process_and_merge_sensor_data_new.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "PydisplaySensorData",
    "SequenceRepairReport",
    "load_hrdata_csv",
    "load_pydisplay_sensor_csv",
    "PYDISPLAY_SENSOR_COLUMNS",
    "QC_COLUMNS",
]

SAMPLE_RATE_HZ: int = 100

PYDISPLAY_SENSOR_COLUMNS: tuple[str, ...] = (
    "frame_seq",
    "absolute_seq_u64",
    "segment_id",
    "sample_seq",
    "PPG_G",
    "PPG_R",
    "PPG_IR",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "GYRO_X",
    "GYRO_Y",
    "GYRO_Z",
    "Uh1",
    "Uh2",
    "Uh3",
    "Uh4",
    "Uc1",
    "Uc2",
    "Uc3",
    "Uc4",
    "UD1",
    "UD2",
    "parser_valid",
)

QC_COLUMNS: tuple[str, ...] = (
    "SampleIndex",
    "Seq",
    "ValidFlag",
    "InterpFlag",
    "GapLen",
    "MissingBefore",
)


@dataclass(frozen=True)
class SequenceRepairReport:
    """Counts recorded while validating and rebuilding a 100 Hz CSV stream."""

    input_rows: int
    output_rows: int
    removed_empty_rows: int
    removed_parser_invalid_rows: int
    removed_duplicate_or_reordered_rows: int
    inserted_missing_rows: int
    frame_seq_mismatches: int
    sample_seq_mismatches: int
    segment_transitions: int


@dataclass(frozen=True)
class PydisplaySensorData:
    """Validated Pydisplay frame plus sequence-repair diagnostics."""

    frame: pd.DataFrame
    report: SequenceRepairReport


def load_pydisplay_sensor_csv(
    sensor_csv: str | Path,
    *,
    fs: int = SAMPLE_RATE_HZ,
) -> PydisplaySensorData:
    """Read the strict Pydisplay CSV and rebuild missing sequence slots."""

    path = Path(sensor_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {path}")
    raw = pd.read_csv(path)
    input_rows = len(raw)
    missing = [name for name in PYDISPLAY_SENSOR_COLUMNS if name not in raw.columns]
    if missing:
        raise KeyError(f"Missing required Pydisplay columns: {', '.join(missing)}")

    empty_mask = raw.loc[:, list(PYDISPLAY_SENSOR_COLUMNS)].isna().all(axis=1)
    removed_empty = int(empty_mask.sum())
    raw = raw.loc[~empty_mask].copy()
    parser_valid = raw["parser_valid"].map(_parser_valid_value).fillna(False).astype(bool)
    removed_invalid = int((~parser_valid).sum())
    raw = raw.loc[parser_valid].copy()
    if raw.empty:
        raise ValueError(f"Sensor CSV has no usable rows after parser_valid filtering: {path}")

    for name in ("frame_seq", "absolute_seq_u64", "segment_id", "sample_seq"):
        values = pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Sequence column {name!r} must contain finite integers: {path}")
        raw[name] = values.astype(np.int64)

    sample_values = raw["sample_seq"].to_numpy(dtype=np.int64)
    sample_mismatches = int(np.sum(np.diff(sample_values) != 1))
    output_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, int | float]] = []
    removed_reordered = 0
    inserted_missing = 0
    frame_mismatches = 0
    segment_transitions = 0
    previous_segment: int | None = None
    previous_absolute: int | None = None
    previous_frame: int | None = None

    for _, series in raw.iterrows():
        row = series.to_dict()
        segment = int(row["segment_id"])
        absolute = int(row["absolute_seq_u64"])
        frame_seq = int(row["frame_seq"])
        if previous_segment is not None and segment != previous_segment:
            segment_transitions += 1
            previous_absolute = None
            previous_frame = None
        gap = 0
        if previous_absolute is not None:
            delta = absolute - previous_absolute
            if delta <= 0:
                removed_reordered += 1
                continue
            gap = delta - 1
            frame_delta = (frame_seq - int(previous_frame)) % 65536
            if frame_delta != delta % 65536:
                frame_mismatches += 1
            for offset in range(1, delta):
                inserted = {name: np.nan for name in PYDISPLAY_SENSOR_COLUMNS}
                inserted["segment_id"] = segment
                inserted["absolute_seq_u64"] = previous_absolute + offset
                inserted["frame_seq"] = (int(previous_frame) + offset) % 65536
                inserted["parser_valid"] = False
                output_rows.append(inserted)
                qc_rows.append(
                    {
                        "Seq": previous_absolute + offset,
                        "ValidFlag": 0,
                        "InterpFlag": 1,
                        "GapLen": gap,
                        "MissingBefore": 0,
                    }
                )
                inserted_missing += 1
        output_rows.append(row)
        qc_rows.append(
            {
                "Seq": absolute,
                "ValidFlag": 1,
                "InterpFlag": 0,
                "GapLen": 0,
                "MissingBefore": gap,
            }
        )
        previous_segment = segment
        previous_absolute = absolute
        previous_frame = frame_seq

    frame = pd.DataFrame(output_rows, columns=PYDISPLAY_SENSOR_COLUMNS)
    qc = pd.DataFrame(qc_rows)
    frame.insert(0, "Time_s", np.arange(len(frame), dtype=float) / float(fs))
    frame.insert(1, "SampleIndex", np.arange(len(frame), dtype=int))
    for name in ("Seq", "ValidFlag", "InterpFlag", "GapLen", "MissingBefore"):
        frame[name] = qc[name].to_numpy()
    report = SequenceRepairReport(
        input_rows=input_rows,
        output_rows=len(frame),
        removed_empty_rows=removed_empty,
        removed_parser_invalid_rows=removed_invalid,
        removed_duplicate_or_reordered_rows=removed_reordered,
        inserted_missing_rows=inserted_missing,
        frame_seq_mismatches=frame_mismatches,
        sample_seq_mismatches=sample_mismatches,
        segment_transitions=segment_transitions,
    )
    return PydisplaySensorData(frame=frame, report=report)


def load_hrdata_csv(hr_csv: str | Path) -> np.ndarray:
    """Read reference HR from the two explicit numeric columns only."""

    path = Path(hr_csv)
    if not path.is_file():
        raise FileNotFoundError(f"HRdata CSV not found: {path}")
    frame = pd.read_csv(path)
    required = ("elapsed_seconds", "hr_bpm")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing required HRdata columns: {', '.join(missing)}")
    elapsed = pd.to_numeric(frame["elapsed_seconds"], errors="coerce").to_numpy(dtype=float)
    bpm = pd.to_numeric(frame["hr_bpm"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(elapsed) & np.isfinite(bpm)
    if int(valid.sum()) < 2:
        raise ValueError(f"HRdata CSV has no usable rows: {path}")
    elapsed = elapsed[valid]
    bpm = bpm[valid]
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError(f"elapsed_seconds must be strictly increasing: {path}")
    return np.column_stack([elapsed, bpm])


def _parser_valid_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and np.isfinite(value):
        return float(value) == 1.0
    return str(value).strip().lower() in {"true", "1"}
