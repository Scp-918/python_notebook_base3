"""Sensor + ground-truth CSV ingestion (Python port of ``process_and_merge_sensor_data_new.m``)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_movmedian_linear,
)

__all__ = [
    "ProcessedDataset",
    "PydisplaySensorData",
    "SequenceRepairReport",
    "load_dataset",
    "load_hrdata_csv",
    "load_pydisplay_sensor_csv",
    "PYDISPLAY_SENSOR_COLUMNS",
    "SENSOR_COLUMNS",
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

# Mapping of internal short name -> raw CSV column header
SENSOR_COLUMNS: dict[str, str] = {
    "Uc1": "Uc1(mV)",
    "Uc2": "Uc2(mV)",
    "Ut1": "Ut1(mV)",
    "Ut2": "Ut2(mV)",
    "PPG_Green": "PPG_Green",
    "PPG_Red": "PPG_Red",
    "PPG_IR": "PPG_IR",
    "AccX": "AccX(g)",
    "AccY": "AccY(g)",
    "AccZ": "AccZ(g)",
    "GyroX": "GyroX(dps)",
    "GyroY": "GyroY(dps)",
    "GyroZ": "GyroZ(dps)",
}

QC_COLUMNS: tuple[str, ...] = (
    "SampleIndex",
    "Seq",
    "ValidFlag",
    "InterpFlag",
    "GapLen",
    "MissingBefore",
)


@dataclass
class ProcessedDataset:
    """Result of :func:`load_dataset`.

    Attributes
    ----------
    data:
        Pandas DataFrame with one row per 10 ms sample. Columns:
        ``Time_s`` plus, for each entry of :data:`SENSOR_COLUMNS`, the
        cleaned raw value and the band-pass-filtered ``<name>_Filt`` value.
    ref_data:
        Two-column ``(N, 2)`` numpy array. Column 0 is the reference time
        in seconds, column 1 is the reference heart-rate in BPM.
    """

    data: pd.DataFrame
    ref_data: np.ndarray


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


def _bandpass_coeffs(
    fs: int = SAMPLE_RATE_HZ,
    low_hz: float = 0.5,
    high_hz: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = fs / 2.0
    high = min(float(high_hz), 0.45 * float(fs))
    low = max(float(low_hz), 1e-3)
    return butter(4, [low / nyquist, high / nyquist], btype="bandpass")


def _bandpass_band_for_column(short: str) -> tuple[float, float]:
    """Return the protocol band for one raw loader channel.

    中文说明：旧 loader 保存的是原始 ``Uc/Ut`` 而不是协议层 ``hf/cf`` 字段。
    这里把冷膜相关通道归到 0.1-5 Hz，PPG 保持 0.5-5 Hz，ACC/Gyro 保持
    0.5-10 Hz，避免回退到 V2 的统一带通。
    """

    if short.startswith("PPG"):
        return 0.5, 5.0
    if short.startswith("Acc") or short.startswith("Gyro"):
        return 0.5, 10.0
    return 0.1, 5.0


def _clean_signal(values: np.ndarray, name: str, fs: int) -> np.ndarray:
    cleaned = fillmissing_nearest(values)
    if "PPG" in name:
        neg = cleaned < 0
        if neg.any():
            cleaned = cleaned.astype(float).copy()
            cleaned[neg] = np.nan
            cleaned = fillmissing_linear(cleaned)
            cleaned = fillmissing_nearest(cleaned)
    return filloutliers_movmedian_linear(cleaned, window=fs)


def _parse_reference_csv(gt_csv: Path) -> np.ndarray:
    # 尝试两种格式：新格式带标准表头，旧格式前3行为元数据。
    candidates: list[pd.DataFrame] = []
    try:
        candidates.append(pd.read_csv(gt_csv))
    except Exception:
        pass
    try:
        candidates.append(pd.read_csv(gt_csv, skiprows=3, header=None))
    except Exception:
        pass

    for gt in candidates:
        if gt.empty:
            continue
        lower_cols = [str(c).strip().lower() for c in gt.columns]
        time_idx = _first_ref_column(lower_cols, ("time", "seconds", "sec", "elapsed"))
        hr_idx = _first_ref_column(lower_cols, ("hr", "heart", "bpm"))

        if time_idx is not None and hr_idx is not None:
            time_s = _parse_ref_time(gt.iloc[:, time_idx])
            bpm = pd.to_numeric(gt.iloc[:, hr_idx], errors="coerce").to_numpy(dtype=float)
        elif gt.shape[1] >= 3:
            time_s = _parse_ref_time(gt.iloc[:, 1])
            bpm = pd.to_numeric(gt.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
        elif gt.shape[1] >= 2:
            time_s = _parse_ref_time(gt.iloc[:, 0])
            bpm = pd.to_numeric(gt.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        else:
            continue

        valid = ~(np.isnan(time_s) | np.isnan(bpm))
        if valid.sum() >= 2:
            return np.column_stack([time_s[valid], bpm[valid]])
    raise ValueError(f"Cannot parse reference CSV: {gt_csv}")


def _first_ref_column(names: list[str], needles: tuple[str, ...]) -> int | None:
    for i, name in enumerate(names):
        if any(n in name for n in needles):
            return i
    return None


def _parse_ref_time(series: pd.Series) -> np.ndarray:
    raw = series.astype(str).str.strip()

    def _to_seconds(t: str) -> float:
        try:
            return pd.to_timedelta(t).total_seconds()
        except (ValueError, TypeError):
            try:
                return float(t)
            except ValueError:
                return float("nan")

    return np.array([_to_seconds(t) for t in raw], dtype=float)


def _time_seconds_from_raw(raw: pd.DataFrame, fs: int) -> np.ndarray:
    """Return sample time, preferring the named ``Time(s)`` column.

    中文说明：新多通道 CSV 已带硬件时间列；旧 CSV 没有时则保留原来的
    ``np.arange(n) / fs`` 行为。
    """

    n = len(raw)
    if "Time(s)" in raw.columns:
        time = pd.to_numeric(raw["Time(s)"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(time).sum() >= max(1, n // 2):
            fallback = np.arange(n, dtype=float) / float(fs)
            mask = np.isfinite(time)
            if not mask.all():
                time = time.copy()
                time[~mask] = fallback[~mask]
            return time
    return np.arange(n, dtype=float) / float(fs)


def _qc_frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Return standard QC columns with defaults for legacy CSV files."""

    n = len(raw)
    defaults: dict[str, np.ndarray] = {
        "SampleIndex": np.arange(n, dtype=int),
        "Seq": np.full(n, np.nan, dtype=float),
        "ValidFlag": np.ones(n, dtype=int),
        "InterpFlag": np.zeros(n, dtype=int),
        "GapLen": np.zeros(n, dtype=int),
        "MissingBefore": np.zeros(n, dtype=int),
    }
    out = pd.DataFrame(index=np.arange(n))
    for column in QC_COLUMNS:
        if column in raw.columns:
            out[column] = pd.to_numeric(raw[column], errors="coerce").to_numpy()
        else:
            out[column] = defaults[column]
    return out


def load_dataset(
    sensor_csv: str | Path,
    gt_csv: str | Path,
    *,
    fs: int = SAMPLE_RATE_HZ,
    columns: Iterable[str] | None = None,
) -> ProcessedDataset:
    """Load and preprocess a sensor + reference CSV pair.

    Parameters
    ----------
    sensor_csv:
        Path to the raw sensor CSV (14-column layout, see :data:`SENSOR_COLUMNS`).
    gt_csv:
        Path to the Polar-style reference CSV (header on rows 1-3, data from row 4).
    fs:
        Target sampling rate (defaults to 100 Hz, matching the original MATLAB pipeline).
    columns:
        Optional iterable of channel short-names to process. ``None`` means all
        13 channels in :data:`SENSOR_COLUMNS`.
    """
    sensor_path = Path(sensor_csv)
    gt_path = Path(gt_csv)
    if not sensor_path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {gt_path}")

    raw = pd.read_csv(sensor_path)
    n = len(raw)
    if n == 0:
        raise ValueError(f"Sensor CSV is empty: {sensor_path}")

    df = pd.DataFrame()
    df["Time_s"] = _time_seconds_from_raw(raw, int(fs))
    qc_frame = _qc_frame_from_raw(raw)
    for column in QC_COLUMNS:
        df[column] = qc_frame[column].to_numpy()

    selected = list(columns) if columns is not None else list(SENSOR_COLUMNS)
    for short in selected:
        if short not in SENSOR_COLUMNS:
            raise KeyError(f"Unknown sensor column '{short}'")
        original = SENSOR_COLUMNS[short]
        if original not in raw.columns:
            raise KeyError(f"Column '{original}' missing in {sensor_path}")
        df[short] = raw[original].astype(float).to_numpy()

    for short in selected:
        cleaned = _clean_signal(df[short].to_numpy(dtype=float), short, fs)
        df[short] = cleaned
        low_hz, high_hz = _bandpass_band_for_column(short)
        b, a = _bandpass_coeffs(fs, low_hz, high_hz)
        df[f"{short}_Filt"] = filtfilt(b, a, cleaned)

    ref_data = _parse_reference_csv(gt_path)
    return ProcessedDataset(data=df, ref_data=ref_data)
